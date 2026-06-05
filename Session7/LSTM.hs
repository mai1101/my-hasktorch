{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

--module Main(main) where
module LSTM where
import Codec.Binary.UTF8.String (encode)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics
import qualified Data.Map.Strict as M 
import Torch
import Data.Char (toLower, ord, chr)
import Data.Word (Word8)
import Data.List (foldl', intersperse, take)
import Control.Monad (when)
import ML.Exp.Chart (drawLearningCurve) 
import System.Mem (performGC)

numIters :: Int
numIters = 80

learnRate :: Tensor
learnRate = asTensor (0.01 :: Float)

wordDimension :: Int 
wordDimension = 256

batchSize :: Int 
batchSize = 32

dataSize :: Int 
dataSize = 1000

lstmHiddenDim :: Int
lstmHiddenDim = 512

-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic, FromJSON, ToJSON)

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
} deriving (Show, Generic, FromJSON, ToJSON)

--------
--spec
--------
data ModelSpec = ModelSpec {
    embeddingSpec :: EmbeddingSpec,
    lstmSpec :: LSTMSpec,
    mlpSpec :: MLPSpec
} deriving (Generic)

data EmbeddingSpec = EmbeddingSpec {
    wordNum :: Int,
    wordDim :: Int
} deriving (Show, Eq, Generic)

data LSTMSpec = LSTMSpec {
    inf :: Int, 
    hf :: Int
} deriving (Show, Eq, Generic)

data MLPSpec = MLPSpec { 
    feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
} deriving (Generic)

--------
--model
--------
-- ★ 解決策：リストをやめて、専用の Gate 型を作りました！
data Gate = Gate {
    weight_ih :: Parameter,
    weight_hh :: Parameter,
    bias_g    :: Parameter
} deriving (Show, Generic, Parameterized)

data LSTM = LSTM { 
    input_gate  :: Gate,
    forget_gate :: Gate,
    output_gate :: Gate,
    hidden_gate :: Gate
} deriving (Show, Generic, Parameterized) -- ★ Parameterized を復活！

data Embedding = Embedding {
    wordEmbedding :: Parameter
} deriving (Show, Generic, Parameterized)

data MLP = MLP { 
    layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
} deriving (Generic, Parameterized)

data Model = Model {
  emb :: Embedding,
  lstmLayer :: LSTM,
  mlp :: MLP
} deriving (Generic, Parameterized)

---------------------
--initialize method
---------------------
instance Randomizable EmbeddingSpec Embedding where
  sample EmbeddingSpec{..} = 
    Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim])

-- Gateごとの初期化を関数化する
sampleGate :: Int -> Int -> IO Gate
sampleGate inf' hf' = do
    w_ih <- makeIndependent =<< randnIO' [hf', inf']
    w_hh <- makeIndependent =<< randnIO' [hf', hf']
    b    <- makeIndependent =<< randnIO' [1, hf']
    return $ Gate w_ih w_hh b

instance Randomizable LSTMSpec LSTM where
  sample LSTMSpec {..} = do
    ig <- sampleGate inf hf
    fg <- sampleGate inf hf
    og <- sampleGate inf hf
    hg <- sampleGate inf hf
    return $ LSTM ig fg og hg

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts 
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) = scanl shift (a, b) t
        where shift (x, y) z = (y, z)

instance Randomizable ModelSpec Model where
    sample ModelSpec {..} = 
        Model <$> sample embeddingSpec <*> sample lstmSpec <*> sample mlpSpec

initialize :: ModelSpec -> FilePath -> IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return Model {emb = loadedEmb , lstmLayer = lstmLayer randomizedModel ,mlp = mlp randomizedModel}

--------------
-- 順伝播の処理
--------------
mlpCalc :: MLP -> Tensor -> Tensor 
mlpCalc MLP {..} input = foldl' (\x f -> f x) input $ intersperse nonlinearity $ map linear layers 

-- ★ Gate型を使ったことで、(!! 0) などの面倒なリスト処理が消滅しました
calcGate :: Tensor -> Tensor -> (Tensor -> Tensor) -> Gate -> Tensor
calcGate input hidden nonLinearity Gate{..} =
  nonLinearity $ (mul input weight_ih) + (mul hidden weight_hh) + (toDependent bias_g)
  where
    mul features wts = transpose2D $ matmul (toDependent wts) (transpose2D features)

newCellState :: LSTM -> Tensor -> Tensor -> Tensor -> Tensor
newCellState LSTM {..} input hidden prevCell =
  (fg * prevCell) + (ig * c')
  where
    ig = calcGate input hidden sigmoid input_gate
    fg = calcGate input hidden sigmoid forget_gate
    c' = calcGate input hidden Torch.tanh hidden_gate

nextState :: LSTM -> Tensor -> Tensor -> Tensor -> Tensor
nextState cell input hidden cNew =
    og * (Torch.tanh cNew)
    where
      og = calcGate input hidden sigmoid (output_gate cell)

lstmCalc :: LSTM -> Tensor -> Tensor
lstmCalc lstmLayer input = 
    let seqLen = size 1 input 
        batch_size = size 0 input 
        wordVectors = map (\i -> select 1 i input) [0 .. seqLen - 1] 
        h_0 = zeros' [batch_size, lstmHiddenDim]
        c_0 = zeros' [batch_size, lstmHiddenDim]
        (h_last, _) = foldl' step (h_0, c_0) wordVectors
        step (prevHidden, prevCell) currentWord = 
            let newCell = newCellState lstmLayer currentWord prevHidden prevCell
                newHidden = nextState lstmLayer currentWord prevHidden newCell
            in (newHidden, newCell)
    in h_last

forWard :: Model -> Tensor -> Tensor
forWard model inputIdxes = 
    let embVector = embedding' (toDependent $ wordEmbedding (emb model)) inputIdxes
        h_last = lstmCalc (lstmLayer model) embVector 
        output = mlpCalc (mlp model) h_last
    in output

--------------------
-- data file の処理
--------------------
amazonReviewPath, amazonReviewPath_v, amazonReviewPath_e, outputPath, embeddingPath, wordLstPath :: FilePath
amazonReviewPath = "Session7/data/train.jsonl"
amazonReviewPath_v = "Session7/data/valid.jsonl"
amazonReviewPath_e = "Session7/data/test.jsonl"
outputPath = "Session7/data/review-texts.txt"
embeddingPath = "Session6/data/sample_embedding.params"
wordLstPath = "Session6/data/sample_wordlst.txt"

loadData :: FilePath -> IO [AmazonReview]
loadData path = do
    jsonl <- B.readFile path
    case decodeToAmazonReview jsonl of
        Left _ -> return []
        Right rev -> return rev

decodeToAmazonReview :: B.ByteString -> Either String [AmazonReview] 
decodeToAmazonReview jsonl = sequenceA $ map eitherDecode (B.split (B.c2w '\n') jsonl)

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "," , "!", "?", ";", ":", "(", ")","*","-"]

toLowerW8 :: Word8 -> Word8
toLowerW8 = fromIntegral . ord . toLower . chr . fromIntegral

prepareTensors :: [AmazonReview] -> (B.ByteString -> Int) -> Int -> (Tensor, Tensor)
prepareTensors reviews wordToIndex padId = 
    let extractAndClean rev = 
            let cleanedtext = B.pack $ map toLowerW8 $ filter (not . isUnncessaryChar) (encode (text rev))
                wordList = Data.List.take 60 $ B.split (head $ encode " ") cleanedtext
                wordIds = map wordToIndex wordList
            in (wordIds, rating rev)
        pairs = map extractAndClean reviews 
        maxLen = maximum (map (length . fst) pairs)
        --padList lst = lst ++ replicate (maxLen - length lst) padId
        padList lst = replicate (maxLen - length lst) padId ++ lst
        paddedX = map (padList . fst) pairs
        listY   = map snd pairs       
    in  (asTensor paddedX, asTensor listY)

wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

-- 未知語の割合を計算する関数
calcUnknownWordRatio :: [AmazonReview] -> (B.ByteString -> Int) -> Int -> Float
calcUnknownWordRatio reviews wordToIndex unkId = 
    let -- 1つのレビューからパディング前の単語リストを抽出
        extractWords rev = 
            let cleanedtext = B.pack $ map toLowerW8 $ filter (not . isUnncessaryChar) (encode (text rev))
                wordList = Data.List.take 60 $ B.split (head $ encode " ") cleanedtext
            in wordList
        
        -- 評価用データの全単語を1つのリストにまとめる
        allWords = concatMap extractWords reviews
        totalWords = length allWords
        
        -- 未知語（wordToIndexの結果がunkIdになる単語）の数をカウント
        unkWords = length $ filter (\w -> wordToIndex w == unkId) allWords
        
    in safeDiv (fromIntegral unkWords) (fromIntegral totalWords)


---------------
-- evaluation
---------------
accuracy :: [Int] -> [Int] -> Float
accuracy y_true y_pred = 
    let correct = length $ filter (==True) $ zipWith (==) y_true y_pred
    in fromIntegral correct / fromIntegral (length y_true) 

countPair :: Int -> Int -> [(Int,Int)] -> Int
countPair i j y_pair = length $ filter (\(t,p) -> p == j && t == i) y_pair

makeConfusionMatrix :: [Int] -> [Int] -> [[Int]]
makeConfusionMatrix y_true y_pred = 
    let classes = [1..5]
        y_pair = zip y_true y_pred
    in [ [ countPair i j y_pair | j <- classes ] | i <- classes ]

-- ゼロ除算を防ぐための安全な割り算ヘルパー関数
safeDiv :: Float -> Float -> Float
safeDiv _ 0.0 = 0.0
safeDiv x y   = x / y

-- F1 score for a given label
calcF1Score :: Int -> [Int] -> [Int] -> Float
calcF1Score target y_true y_pred = 
    let p = precision target y_true y_pred 
        r = recall target y_true y_pred 
    in safeDiv (2 * p * r) (p + r)

-- Macro F1 score
macroF1Score :: Int -> [Int] -> [Int] -> Float
macroF1Score n y_true y_pred = 
    let classes = [1..n]  -- ★ 0..(n-1) から 1..n に修正 (星1〜5のため)
        f1Scores = [calcF1Score m y_true y_pred | m <- classes]
    in (sum f1Scores) / (fromIntegral n) 

-- Weighted F1 score
weightedF1Score :: Int -> [Int] -> [Int] -> Float
weightedF1Score n y_true y_pred = 
    let classes = [1..n]  -- ★ ここも 1..n に修正
        f1Scores = [calcF1Score m y_true y_pred | m <- classes]
        supports = [fromIntegral $ countSupport t y_true | t <- classes]
        totalLen = fromIntegral $ length y_true 
        weights = map (\s -> s / totalLen) supports
    in sum $ zipWith (*) f1Scores weights

-- Count the number of occurrences of “target”
countSupport :: Int -> [Int] -> Int
countSupport target y_true = length $ filter (== target) y_true

-- Micro F1 score
microF1Score :: Int -> [Int] -> [Int] -> Float
microF1Score n y_true y_pred = 
    let classes = [1..n]  -- ★ ここも 1..n に修正
        y_pair = zip y_true y_pred
        total_TP = fromIntegral $ sum [countTP m y_pair | m <- classes]
        total_FP = fromIntegral $ sum [countFP m y_pair | m <- classes]
        total_FN = fromIntegral $ sum [countFN m y_pair | m <- classes]
    in safeDiv total_TP (total_TP + 0.5 * (total_FP + total_FN))

countTP :: Int -> [(Int,Int)] -> Int
countTP target y_pair = length $ filter (\(t,p) -> t == target && p == target) y_pair 

countFP :: Int -> [(Int,Int)] -> Int
countFP target y_pair = length $ filter (\(t,p) -> t /= target && p == target) y_pair

countFN :: Int -> [(Int,Int)] -> Int
countFN target y_pair = length $ filter (\(t,p) -> t == target && p /= target) y_pair

-- precision function
precision :: Int -> [Int] -> [Int] -> Float
precision target y_true y_pred = 
    let y_pair = zip y_true y_pred
        predictTarget = filter (\(_,p) -> p == target) y_pair
        truePositive = filter (\(t,_) -> t == target) predictTarget
        len_tp = fromIntegral $ length truePositive
        len_target = fromIntegral $ length predictTarget
    in safeDiv len_tp len_target -- ★ 0で割るのを防ぐ

-- recall function
-- The proportion of correctly predicted 'A' cases out of all actual 'A' cases. (コメント修正)
recall :: Int -> [Int] -> [Int] -> Float
recall target y_true y_pred = 
    let y_pair = zip y_true y_pred
        haveTarget = filter (\(t,_) -> t == target) y_pair
        truePositive = filter (\(_,p) -> p == target) haveTarget
        len_tp = fromIntegral $ length truePositive
        len_target = fromIntegral $ length haveTarget
    in safeDiv len_tp len_target -- ★ 0で割るのを防ぐ


----------
--main
----------
main :: IO ()
main = do
    wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
    
    let wordToIndex = wordToIndexFactory wordLst
        wordNumValue = length wordLst + 1
        initEmbSpec = EmbeddingSpec {wordNum = wordNumValue, wordDim = wordDimension}    
        initMlpSpec = MLPSpec {feature_counts = [lstmHiddenDim, 32, 1], nonlinearitySpec = Torch.tanh}
        initLstmSpec = LSTMSpec {inf = wordDimension, hf = lstmHiddenDim}
        modelSpec = ModelSpec { embeddingSpec = initEmbSpec, lstmSpec = initLstmSpec, mlpSpec = initMlpSpec }

    initModel <- sample modelSpec
    --initModel <- initialize modelSpec embeddingPath
    putStrLn "finish initialize"
    
    reviews   <- loadData amazonReviewPath
    reviews_v <- loadData amazonReviewPath_v
    reviews_e <- loadData amazonReviewPath_e
    putStrLn "finish load data"

    let unkId = length wordLst
        smallReviews = Prelude.take dataSize reviews 
        smallReviews_v = Prelude.take (dataSize `Prelude.div` 8) reviews_v
        smallReviews_e = Prelude.take (dataSize `Prelude.div` 8) reviews_e
        (xs_valid, ys_valid) = prepareTensors smallReviews_v wordToIndex unkId 
        (xs_eval,  ys_eval ) = prepareTensors smallReviews_e wordToIndex unkId 
    putStrLn "finish prepare data"

    (trained, lossList) <- foldLoop (initModel, []) numIters $ \(state, pastLossList) i -> do
        let startIdx = (i * batchSize) `mod` (length smallReviews - batchSize)
            batchReviews = Data.List.take batchSize $ drop startIdx smallReviews
            (batch_x, batch_y) = prepareTensors batchReviews wordToIndex unkId 

        let y  = reshape [-1, 1] batch_y          
            y' = forWard state batch_x 
            loss = mseLoss y y'  
        
        -- for validation
        let loss_valid = mseLoss (reshape [-1, 1] ys_valid) (forWard state xs_valid) 
            currentLoss = asValue loss_valid :: Float    


        when (i `mod` 10 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss_valid: " ++ show loss_valid ++ " | Loss_train: " ++ show loss
            performGC

        (newState, _) <- runStep state GD loss learnRate 
        
        currentLoss `seq` return (newState, currentLoss : pastLossList) 
        
    let finalLosses = reverse lossList
    drawLearningCurve "Session7/loss_decrease.png" "Learning Curve" [("loss", finalLosses)]

    -------------
    -- evalation
    -------------

    let y_pred_tensor = reshape [-1] $ forWard trained xs_eval
        y_true = asValue ys_eval :: [Float]
        y_pred = asValue y_pred_tensor :: [Float]
        y_true_int = map round y_true :: [Int]
        y_pred_int = map round y_pred :: [Int]
        acc = accuracy y_true_int y_pred_int
        confMatrix = makeConfusionMatrix y_true_int y_pred_int
        microf1 = microF1Score 5 y_true_int y_pred_int
        macrof1 = macroF1Score 5 y_true_int y_pred_int
        weightf1 = weightedF1Score 5 y_true_int y_pred_int

        unkRatio = calcUnknownWordRatio smallReviews_e wordToIndex unkId --辞書にないワードのカウント

    putStrLn $ "\n=== Final Evaluation ==="
    putStrLn $ "Accuracy: " ++ show (acc * 100) ++ "%"
    putStrLn $ "Micro F1:    " ++ show microf1
    putStrLn $ "Macro F1:    " ++ show macrof1
    putStrLn $ "Weighted F1: " ++ show weightf1

    putStrLn $ "Unknown Word Ratio: " ++ show (unkRatio * 100) ++ "%"
    putStrLn "Confusion Matrix (Row: True, Col: Pred):"
    mapM_ print confMatrix

    return ()