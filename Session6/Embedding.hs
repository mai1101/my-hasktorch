{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Embedding where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import Data.Char (toLower, ord, chr)
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List
import Control.Monad (when, forM_)
import ML.Exp.Chart (drawLearningCurve)  
import Torch

numIters :: Int
numIters = 2000

learnRate :: Tensor
learnRate = asTensor (0.2 :: Float)

wordDimension :: Int
wordDimension = 9

batchSize :: Int
batchSize = 64

dataSize :: Int
dataSize = 3000


-- your text data (try small data first)
textFilePath = "Session6/data/sample.txt"
modelPath =  "Session6/data/sample_embedding.params"
wordLstPath = "Session6/data/sample_wordlst.txt"

------------
--MLP
------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],             -- neuron list
    nonlinearitySpec :: Tensor -> Tensor -- activation function
  }

data MLP = MLP
  { layers :: [Linear],                  -- weight and bias list
    nonlinearity :: Tensor -> Tensor     -- activation function
  }
  deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where -- initialize layer
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts 
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) = scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor 
mlp MLP {..} input = foldl' (\x f -> f x) input $ intersperse nonlinearity $ map linear layers 

-- take model and id tensor then return prediction
forWard :: Model -> Tensor -> Tensor
forWard Model{..} inputIdxes = 
  -- take vector from Embedding layer
  let embVector = embedding' (toDependent $ wordEmbedding embeddings) inputIdxes
  -- pass it mlp
      prediction = mlp mlpLayer embVector
  in logSoftmax (Dim 1) prediction


--------------
--enbedding
--------------
data EmbeddingSpec = EmbeddingSpec {
    wordNum :: Int, -- the number of words 
    wordDim :: Int  -- the dimention of word embeddings 
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter -- weight
  } deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model {
    mlpLayer :: MLP,
    embeddings :: Embedding
  } deriving (Generic, Parameterized)

-- for remove unnecessary letter
isUnncessaryChar :: 
    Word8 ->
    Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "," , "!", "?", ";", ":", "(", ")","*","-"]

-- change the all letter of word to small letter
toLowerW8 :: Word8 -> Word8
toLowerW8 = fromIntegral . ord . toLower . chr . fromIntegral

--change "This is a pen." to list like ["This", "is", "a", "pen"] 
preprocess ::
    B.ByteString -> -- input
    [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ map toLowerW8 $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
    [B.ByteString] ->     -- wordlist
    (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

toyEmbedding ::
    EmbeddingSpec ->
    Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim


-- return (Input, Target) 
prepareTensors :: [[B.ByteString]] -> (B.ByteString -> Int) -> (Tensor, Tensor)
prepareTensors wordLines wordToIndex = 
    let idxes  = map (map wordToIndex) wordLines
        pairs = concatMap makeAllPairs idxes
    in (asTensor $ map fst pairs, asTensor $ map snd pairs)


-- for skipgram /  return (centor, around centor)
makeAllPairs :: [Int] -> [(Int, Int)]
makeAllPairs [] = []
makeAllPairs [id] = []
makeAllPairs (id1 : id2 : ids) = (id1,id2) : (id2,id1) : makeAllPairs (id2 : ids)

splitData :: [a] -> ([a],[a])
splitData xs = 
    let n = length xs
        n_train = (n * 9) `Prelude.div` 10
        (train, valid) = splitAt n_train xs
    in (train, valid)

main :: IO ()
main = do
    -- load text file
  
    texts <- B.readFile textFilePath
    -- Create a unique word list
    let allwordLines = preprocess texts 
        wordLines = Data.List.take dataSize allwordLines
        (wordLines_train, wordLines_valid) = splitData wordLines
        wordlst = nub $ concat wordLines_train   -- List of unique words
        wordToIndex = wordToIndexFactory wordlst -- wordToIndex: pass word then return id
        wordNumValue = length wordlst + 1


    let idxes_train = map (map wordToIndex) wordLines_train
        pairs_train = concatMap makeAllPairs idxes_train --  (Int, Int) 
        totalPairs  = length pairs_train
        --(xs_train, ys_train) = prepareTensors wordLines_train wordToIndex    -- it can use when full-batch
        (xs_valid, ys_valid) = prepareTensors wordLines_valid wordToIndex


     -- Create initial embedding (wordDim × wordNum) -
    let embeddingSpec = EmbeddingSpec {wordNum = wordNumValue, wordDim = wordDimension}
    wordEmb <- makeIndependent $ toyEmbedding embeddingSpec 
    print "d"
    let initEmb = Embedding { wordEmbedding = wordEmb }  --Embedding data
    
    -- initialize the model
    initMlp <- sample $ MLPSpec [wordDimension, 10, wordNumValue] Torch.tanh
    let initModel = Model initMlp initEmb

    (trained, lossList) <- foldLoop (initModel, []) numIters $ \(state, pastLossList) i -> do
        -- calculate Which data to use from batch-size
        let startIdx = (i * batchSize) `mod` (totalPairs - batchSize)

        --  take batch-size pair of (input,target) and change it tensor
            batchPairs = Data.List.take batchSize $ drop startIdx pairs_train
            batch_x = asTensor $ map fst batchPairs
            batch_y = asTensor $ map snd batchPairs

        -- calculate loss
        let y  = batch_y                     
            y' = forWard state batch_x            
            loss = nllLoss' y y'  
      {-
        -- (its code use for only full-batch approrch)
        let y  = ys_train                          -- y: true value
            y' = forWard state xs_train            -- y':estimated value
            loss = nllLoss' y y'
      -}
        -- for validation
        let loss_valid = nllLoss' ys_valid (forWard state xs_valid) 
            currentLoss = asValue loss_valid :: Float   

        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss_valid: " ++ show loss_valid ++ "| Loss_train: " ++ show loss

        (newState, _) <- runStep state GD loss learnRate 

        currentLoss `seq` return (newState, currentLoss : pastLossList) 
    

    let finalLosses = reverse lossList
    drawLearningCurve "Session6/loss_decrease1.png" "Learning Curve" [("loss", finalLosses)]


    -- Save params to use trained parameter in the next session
    saveParams (embeddings trained) modelPath
    -- Save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
    
    --2
    -- initialize
    dummyWeights <- makeIndependent $ zeros' [wordNumValue, wordDimension] 
    let testInitEmb = Embedding { wordEmbedding = dummyWeights }
    -- load file
    loadedEmb <- loadParams testInitEmb modelPath
    -- take vector of "it"
    let testWord = B.pack $ encode "it"
        testId = wordToIndex testWord
        testEmbTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor [testId])
    putStrLn $ "it : " ++ show testEmbTxt




    --4
    let testWord1 = B.pack $ encode "pc"
        id_pc = wordToIndex testWord1
        testWord2 = B.pack $ encode "screen"
        id_screen = wordToIndex testWord2
        testWord3 = B.pack $ encode "monitor"
        id_monitor = wordToIndex testWord3  
        v_pc = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor [id_pc])
        v_screen  = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor [id_screen])
        v_monitor   = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor [id_monitor])

    -- 意味の足し算・引き算
    putStrLn $ "monitor : " ++ show v_monitor
    let resultVector = v_pc + v_screen
    putStrLn $ "pc + screen : " ++ show resultVector




    return ()