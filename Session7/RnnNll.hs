{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


--module RnnNll where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics
import qualified Data.Map.Strict as M 

{-
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO')
import Torch.Autograd (makeIndependent)
import Torch.Tensor (Tensor, asTensor)
import Torch.Functional (embedding', tanh)
import Torch.TensorFactories (eye', zeros')
-}
import Torch
import Data.Char (toLower, ord, chr)
import Data.Word (Word8)
import Data.List (foldl', intersperse, take)
import Control.Monad (when)

import ML.Exp.Chart (drawLearningCurve) 

numIters :: Int
numIters = 1000

learnRate :: Tensor
learnRate = asTensor (0.002 :: Float)

wordDimension :: Int 
wordDimension = 64

batchSize :: Int 
batchSize = 32

dataSize :: Int -- number of using reviews for train_data
dataSize = 1000

rnnHiddenDim :: Int
rnnHiddenDim = 64  



-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview



--------
--spec
--------

-- definition of model spec
data ModelSpec = ModelSpec {
    embeddingSpec :: EmbeddingSpec,
    rnnSpec :: RNNSpec,
    mlpSpec :: MLPSpec
} deriving (Generic)

data EmbeddingSpec = EmbeddingSpec {
    wordNum :: Int, -- the number of words 
    wordDim :: Int  -- the dimention of word embeddings 
} deriving (Show, Eq, Generic)

data RNNSpec = RNNSpec { -- 参考；Elman.hs
    in_features :: Int, 
    hidden_features :: Int
} deriving (Show, Eq, Generic)

data MLPSpec = MLPSpec { 
    feature_counts :: [Int],             -- neuron list
    nonlinearitySpec :: Tensor -> Tensor -- activation function
} deriving (Generic)


--------
--model
--------
-- definition of model structure
data Model = Model {
  emb :: Embedding,
  -- TODO: add RNN
  rnn :: RNN,
  mlp :: MLP
} deriving (Generic, Parameterized)

-- definition of embedding layer
data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

data RNN = RNN { -- 参考：Elman.hs
    input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
} deriving (Show, Generic, Parameterized)

data MLP = MLP
  { layers :: [Linear],                  -- weight and bias list
    nonlinearity :: Tensor -> Tensor     -- activation function
  }
  deriving (Generic, Parameterized)





---------------------
--initialize mothod
---------------------

instance Randomizable EmbeddingSpec Embedding where
  sample EmbeddingSpec{..} = 
    Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim])

-- 参考：Elman.hs
instance Randomizable RNNSpec RNN where
  sample RNNSpec {..} = do
    w_ih <- makeIndependent =<< randnIO' [in_features, hidden_features]
    w_hh <- makeIndependent =<< randnIO' [hidden_features, hidden_features]
    b <- makeIndependent =<< randnIO' [1, hidden_features]
    return $ RNN w_ih w_hh b

instance Randomizable MLPSpec MLP where -- initialize layer
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts 
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) = scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

-- sample : how to initialize
instance Randomizable ModelSpec Model where
    sample ModelSpec {..} = 
        Model
        <$> sample embeddingSpec 
        -- TODO: add RNN initilization
        <*> sample rnnSpec
        <*> sample mlpSpec

-- randomize and initialize embedding with loaded params
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return Model {emb = loadedEmb , rnn = rnn randomizedModel ,mlp = mlp randomizedModel}


--------------
-- 順伝播の処理
--------------

mlpCalc :: MLP -> Tensor -> Tensor 
mlpCalc MLP {..} input = foldl' (\x f -> f x) input $ intersperse nonlinearity $ map linear layers 

-- calculate  h_t = tanh(W_ih * x_t + W_hh * h_t-1 + b) (参考：Elman.hs)
nextState :: RNN -> Tensor -> Tensor -> Tensor
nextState RNN {..} input hidden = 
    let w_ih = toDependent input_weight
        w_hh = toDependent hidden_weight
        b    = toDependent bias
    in Torch.tanh (Torch.matmul input w_ih + Torch.matmul hidden w_hh + b)

rnnCalc :: RNN -> Tensor -> Tensor
rnnCalc rnnLayer input = 
    let seqLen = size 1 input -- length of sentence
        batch_size = size 0 input  --batchsize
        wordVectors = map (\i -> select 1 i input) [0 .. seqLen - 1]
        -- first hidden
        h_0 = zeros' [batch_size, rnnHiddenDim]
        -- foldl'  (\前の記憶 今の単語 -> 新しい記憶) 前の記憶　単語の列
        h_last = foldl' (\prevHidden currentWord -> nextState rnnLayer currentWord prevHidden) h_0 wordVectors
    in h_last


--  forWard function
forWard :: Model -> Tensor -> Tensor
forWard model inputIdxes = 
    let embVector = embedding' (toDependent $ wordEmbedding (emb model)) inputIdxes
        h_last = rnnCalc (rnn model) embVector 
        output = mlpCalc (mlp model) h_last
    in Torch.logSoftmax (Dim 1) output --for nll



--------------------
-- data file の処理
--------------------

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "Session7/data/train.jsonl"

amazonReviewPath_v :: FilePath
amazonReviewPath_v = "Session7/data/valid.jsonl"

amazonReviewPath_e :: FilePath
amazonReviewPath_e = "Session7/data/test.jsonl"

outputPath :: FilePath
outputPath = "Session7/data/review-texts.txt"

embeddingPath =  "Session6/data/sample_embedding.params"

wordLstPath = "Session6/data/sample_wordlst.txt"

loadData :: FilePath -> IO [AmazonReview]
loadData path = do
    jsonl <- B.readFile path
    let amazonReviews = decodeToAmazonReview jsonl
    let reviews = case amazonReviews of
                    Left err -> []
                    Right rev -> rev
    return reviews

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList



-- for remove unnecessary letter
isUnncessaryChar :: 
    Word8 ->
    Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "," , "!", "?", ";", ":", "(", ")","*","-"]

-- change the all letter of word to small letter
toLowerW8 :: Word8 -> Word8
toLowerW8 = fromIntegral . ord . toLower . chr . fromIntegral

-- return (Input tensor, Target tensor) 
prepareTensors :: [AmazonReview] -> (B.ByteString -> Int) -> Int -> (Tensor, Tensor)
prepareTensors reviews wordToIndex padId = 
    let -- change one review to (input,target) pair
        -- extractAndClean :: AmazonReview -> ([Int], Int)
        extractAndClean rev = 
            let cleanedtext = B.pack $ map toLowerW8 $ filter (not . isUnncessaryChar) (encode (text rev))
                --wordList = B.split (head $ encode " ") cleanedtext  
                wordList = Data.List.take 30 $ B.split (head $ encode " ") cleanedtext    -- delete long sentence                        
                wordIds = map wordToIndex wordList
                y = round (rating rev) - 1 :: Int --nllLossの場合
            in (wordIds, y)
        
        -- 全てのレビューに extractAndClean を適用 -> [(input=単語Idのリスト,output=score), (input,output), ...]
        pairs = map extractAndClean reviews 
        
        -- padding
        maxLen = maximum (map (length . fst) pairs) --max length of sentence
        padList lst = replicate (maxLen - length lst) padId ++ lst
        
        paddedX = map (padList . fst) pairs
        listY   = map snd pairs       
    in  (asTensor paddedX, asTensor listY)


wordToIndexFactory ::
    [B.ByteString] ->     -- wordlist
    (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))


---------------
-- evaluation
---------------
-- take two label's list then return accuracy
accuracy :: [Int] -> [Int] -> Float
accuracy y_true y_pred = 
    let correct = length $ filter (==True) $ zipWith (==) y_true y_pred
        len = length y_true
    in (fromIntegral correct) / (fromIntegral len) 

countPair :: Int -> Int -> [(Int,Int)] -> Int
countPair i j y_pair = 
    let true_i_pred_j = filter  (\(t,p) -> p == j && t == i) y_pair
    in length true_i_pred_j

makeConfusionMatrix :: [Int] -> [Int] -> [[Int]]
makeConfusionMatrix y_true y_pred = 
    let classes = [1..5]
        y_pair = zip y_true y_pred
    in [ [ countPair i j y_pair | j <- classes ] | i <- classes ]


main :: IO ()
main = do
    -- load word list (It's important to use the same list as whan creating embeddings)
    wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
    
    let wordToIndex = wordToIndexFactory wordLst -- wordToIndex: take wordlist then return id
        wordNumValue = length wordLst + 1

    -- Create initial model
    let initEmbSpec = EmbeddingSpec {wordNum = wordNumValue, wordDim = wordDimension}    
        initRnnSpec = RNNSpec {in_features = wordDimension, hidden_features = rnnHiddenDim}
        initMlpSpec = MLPSpec {feature_counts = [rnnHiddenDim, 10, 5], nonlinearitySpec = Torch.tanh} 
        modelSpec = ModelSpec { embeddingSpec = initEmbSpec, rnnSpec = initRnnSpec, mlpSpec = initMlpSpec }
    initModel <- sample modelSpec --1.b random initialize
    --initModel <- initialize modelSpec embeddingPath --1.c
    
    -- load data
    reviews   <- loadData amazonReviewPath   --train      
    reviews_v <- loadData amazonReviewPath_v --valid
    reviews_e <- loadData amazonReviewPath_e --eval

    -- prepare input tensor and target tensor
    let unkId = length wordLst

        smallReviews = Prelude.take dataSize reviews 
        smallReviews_v = Prelude.take (dataSize `Prelude.div` 8) reviews_v
        smallReviews_e = Prelude.take (dataSize `Prelude.div` 8) reviews_e
        (xs_valid, ys_valid) = prepareTensors smallReviews_v wordToIndex unkId 
        (xs_eval,  ys_eval ) = prepareTensors smallReviews_e wordToIndex unkId 

    -- train!  
    (trained, lossList) <- foldLoop (initModel, []) numIters $ \(state, pastLossList) i -> do
        
        -- calculate Which data to use from batch-size
        let startIdx = (i * batchSize) `mod` (length smallReviews - batchSize)

        --  take batch-size pair of (input,target) and change it tensor
            batchReviews = Data.List.take batchSize $ drop startIdx smallReviews
            (batch_x, batch_y) = prepareTensors batchReviews wordToIndex unkId 

        -- calculate loss
        let y  = batch_y             
            y' = forWard state batch_x 
            loss = nllLoss' y y' 

        -- for validation
        let loss_valid = nllLoss' ys_valid (forWard state xs_valid) 
            currentLoss = asValue loss_valid :: Float    

        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss_valid: " ++ show loss_valid ++ "| Loss_train: " ++ show loss

        (newState, _) <- runStep state GD loss learnRate 

        currentLoss `seq` return (newState, currentLoss : pastLossList) 

    let finalLosses = reverse lossList
    drawLearningCurve "Session7/loss_decrease.png" "Learning Curve" [("loss", finalLosses)]


    -------------
    -- evalation
    -------------
    let --モデルの出力 [N, 5] から、一番確率が高い箱のインデックス(0〜4)を取得する
        y_pred_idx = argmax (Dim 1) RemoveDim (forWard trained xs_eval)
        
        y_true_list = asValue ys_eval    :: [Int]
        y_pred_list = asValue y_pred_idx :: [Int]

        y_true_int = map (+1) y_true_list
        y_pred_int = map (+1) y_pred_list
        
        acc = accuracy y_true_int y_pred_int
        confMatrix = makeConfusionMatrix y_true_int y_pred_int

    -- print result
    putStrLn $ "\n=== Final Evaluation ==="
    putStrLn $ "Accuracy: " ++ show (acc * 100) ++ "%"
    putStrLn "Confusion Matrix (Row: True, Col: Pred):"
    mapM_ print confMatrix
    print (forWard trained xs_eval)

    return ()