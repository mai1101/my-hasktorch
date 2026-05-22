module Evaluate where

import qualified Embedding as E
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BC 
import Codec.Binary.UTF8.String (encode)
import Control.Monad (forM_, when)
import Data.List(take)
import Torch

-- file path
stsFilePath = "Session6/data/answer-answer.test.tsv" 
modelPath =  "Session6/data/sample_embedding_3000.params"
wordLstPath = "Session6/data/sample_wordlst_3000.txt"


parseTSV :: B.ByteString -> [[B.ByteString]]
parseTSV content = 
    let -- split by（\n）
        allLines = B.split (head $ encode "\n") content
        -- split by（\t）
        parsedLines = map (B.split (head $ encode "\t")) allLines
        -- eliminate unnnecesary row
        validLines = filter (\line -> length line >= 3 && not (B.null (head line))) parsedLines
    in validLines

-- change "This is a pen."  to ["This", "is", "a", "pen"] 
preprocess' ::
    B.ByteString -> -- input
    [B.ByteString]  -- wordlist per line
preprocess' texts = 
    let filteredtexts = B.pack $ map E.toLowerW8 $ filter (not . E.isUnncessaryChar) (B.unpack texts)
    in B.split (head $ encode " ") filteredtexts

-- change word list to one vector of sentence
sentenceToVector :: 
    E.Embedding ->             
    (B.ByteString -> Int) ->  
    [B.ByteString] ->         
    Tensor
sentenceToVector emb wordToIndex wordss = 
    let ids = asTensor $ map wordToIndex wordss
        wordVecs = embedding' (toDependent $ E.wordEmbedding emb) ids
    in meanDim (Dim 0) KeepDim Float wordVecs

-- change float value to int label
discretize :: Float -> Int
discretize v
    | v > 0.998 = 5  
    | v > 0.990 = 4  
    | v > 0.980 = 3 
    | v > 0.900 = 2  
    | v > 0.800 = 1  
    | otherwise = 0  

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

makeConfusionMatrix :: Int -> [Int] -> [Int] -> [[Int]]
makeConfusionMatrix n y_true y_pred = 
    let classes = [0..(n-1)]
        y_pair = zip y_true y_pred
    in [ [ countPair i j y_pair | j <- classes ] | i <- classes ]


main :: IO ()
main = do
    stsData <- B.readFile stsFilePath
    let parsedData = parseTSV stsData

    -- load wordlist
    wordLstData <- B.readFile wordLstPath
    let loadedWordLst = B.split (head $ encode "\n") wordLstData
        wordNumValue = length loadedWordLst + 1
        wordToIndex = E.wordToIndexFactory loadedWordLst

    -- make embegging box
    dummyWeights <- makeIndependent $ zeros' [wordNumValue, 9] 
    let testInitEmb = E.Embedding { E.wordEmbedding = dummyWeights }

    -- load file
    loadedEmb <- loadParams testInitEmb modelPath
    
    predictLabels <- foldLoop [] ((length parsedData)-1) $ \currentLabels i -> do
        let line = parsedData !! i
            score = head line
            sent1 = line !! 1
            sent2 = line !! 2
            
        let wordsOfSent1 = preprocess' sent1
            wordsOfSent2 = preprocess' sent2

            vecOfSent1 = sentenceToVector loadedEmb wordToIndex wordsOfSent1
            vecOfSent2 = sentenceToVector loadedEmb wordToIndex wordsOfSent2

            similarity = cosineSimilarity' vecOfSent1 vecOfSent2
            simValue = asValue similarity :: Float
            label = discretize simValue
        
        -- print only first 10 row
        when (i < 11) $ do
            putStrLn $ "(" ++ show i ++ ")"
            putStrLn $ "[Sentence 1 ]: " ++ BC.unpack sent1
            putStrLn $ "[Sentence 2 ]: " ++ BC.unpack sent2
            putStrLn $ "[cosineSimilarity]: " ++ show simValue
            putStrLn $ "[Human Score]: " ++ BC.unpack score ++ "  |  [EstimatedScore]: " ++ show label
            putStrLn ""
        return (label : currentLabels)

    let correctLabels = [ read (BC.unpack $ head line) :: Int | line <- parsedData ]
        acc = accuracy correctLabels (reverse predictLabels)
        confusionMatrix = makeConfusionMatrix 6 correctLabels (reverse predictLabels)
    putStrLn $ "... "
    putStrLn $ "accuracy: " ++ show acc 
    putStrLn $ "confusionMatrix: " ++ show confusionMatrix
    
