{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

import Control.Monad (when, forM, forM_, (<=<))
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import ML.Exp.Chart (drawLearningCurve)  
import Data.Csv (decode, HasHeader(HasHeader), FromRecord)
import qualified Data.Vector as V
import qualified Data.ByteString.Lazy as BL

import Control.Exception.Safe (SomeException (..), try)
import Control.Monad.Cont (ContT (..))
import Pipes
import qualified Pipes.Prelude as P
import Torch.Typed.Vision (initMnist)
import qualified Torch.Vision as V
import Prelude hiding (exp)

-- hyperparameter
epoch :: Int
epoch = 5    -- how many times to learn

learnRate :: Tensor
learnRate = 1e-3      -- learning rate

actFunc :: Tensor -> Tensor
actFunc = Torch.tanh   -- activation function


-- 1バッチごとに学習を回すループ
trainLoop :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> ListT IO (Tensor, Tensor) -> IO (MLP, [Float])
trainLoop validDataList currentModel optimizer = P.foldM step begin done . enumerateData
  where
    step :: (MLP, [Float]) -> ((Tensor, Tensor), Int) -> IO (MLP, [Float])
    step (m, pastValidLosses) ((input, label), iter) = do
      let y' = logSoftmax (Dim 1) (model m input)
          loss = nllLoss' label y'
      (newModel, _) <- runStep m optimizer loss learnRate
      vLoss <- calcValidLoss newModel validDataList
      
      when (iter `mod` 200 == 0) $ do
        let trainLossVal = asValue loss :: Float
        putStrLn $ "  Iteration: " ++ show iter ++ " | Train Loss: " ++ show trainLossVal ++ " | Valid Loss: " ++ show vLoss
        
      pure (newModel, vLoss : pastValidLosses)
      
    begin = pure (currentModel, [])
    done (m, vLosses) = pure (m, reverse vLosses)


calcValidLoss :: MLP -> [(Tensor, Tensor)] -> IO Float
calcValidLoss m validDataList = do
  losses <- forM validDataList $ \(img, label) -> do
    let y' = logSoftmax (Dim 1) (model m img)
        loss = nllLoss' label y'
    return (asValue loss :: Float)
  return $ sum losses / fromIntegral (length validDataList)


--------------
-- evaluation
--------------

evaluateModel :: MLP -> V.MNIST IO -> IO ()
evaluateModel m testMnist = do
  -- ここでは評価用データの先頭100件を使ってスコアを出します
  let evalSize = 100
      startIdx = 100 
  
  pairs <- forM [startIdx .. (startIdx + evalSize - 1)] $ \idx -> do
    (testImg, testLabel) <- getItem testMnist idx
    let output = model m testImg
        predictionTensor = argmax (Dim 1) RemoveDim output
        predVal = asValue predictionTensor :: Int
        trueVal = asValue testLabel :: Int
    return (trueVal, predVal)

  let y_true = map fst pairs
      y_pred = map snd pairs

  let acc = accuracy y_true y_pred
      confMatrix = makeConfusionMatrixMNIST y_true y_pred
      macrof1 = macroF1ScoreMNIST 10 y_true y_pred

  putStrLn $ "\n=== MNIST Final Evaluation Results (Pure Test Data) ==="
  putStrLn $ "Accuracy: " ++ show (acc * 100) ++ "%"
  putStrLn $ "Macro F1: " ++ show macrof1
  putStrLn "\nConfusion Matrix (Row: True, Col: Pred) 0-9:"
  mapM_ print confMatrix


accuracy :: [Int] -> [Int] -> Float
accuracy y_true y_pred = 
    let correct = length $ filter (==True) $ zipWith (==) y_true y_pred
    in fromIntegral correct / fromIntegral (length y_true) 

countPair :: Int -> Int -> [(Int,Int)] -> Int
countPair i j y_pair = length $ filter (\(t,p) -> p == j && t == i) y_pair

safeDiv :: Float -> Float -> Float
safeDiv _ 0.0 = 0.0
safeDiv x y   = x / y

calcF1Score :: Int -> [Int] -> [Int] -> Float
calcF1Score target y_true y_pred = 
    let p = precision target y_true y_pred 
        r = recall target y_true y_pred 
    in safeDiv (2 * p * r) (p + r)

precision :: Int -> [Int] -> [Int] -> Float
precision target y_true y_pred = 
    let y_pair = zip y_true y_pred
        predictTarget = filter (\(_,p) -> p == target) y_pair
        truePositive = filter (\(t,_) -> t == target) predictTarget
        len_tp = fromIntegral $ length truePositive
        len_target = fromIntegral $ length predictTarget
    in safeDiv len_tp len_target

recall :: Int -> [Int] -> [Int] -> Float
recall target y_true y_pred = 
    let y_pair = zip y_true y_pred
        haveTarget = filter (\(t,_) -> t == target) y_pair
        truePositive = filter (\(_,p) -> p == target) haveTarget
        len_tp = fromIntegral $ length truePositive
        len_target = fromIntegral $ length haveTarget
    in safeDiv len_tp len_target

makeConfusionMatrixMNIST :: [Int] -> [Int] -> [[Int]]
makeConfusionMatrixMNIST y_true y_pred = 
    let classes = [0..9]
        y_pair = zip y_true y_pred
    in [ [ countPair i j y_pair | j <- classes ] | i <- classes ]

macroF1ScoreMNIST :: Int -> [Int] -> [Int] -> Float
macroF1ScoreMNIST n y_true y_pred = 
    let classes = [0..(n-1)]
        f1Scores = [calcF1Score m y_true y_pred | m <- classes]
    in (sum f1Scores) / (fromIntegral n)

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where
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

model :: MLP -> Tensor -> Tensor
model params t = mlp params t


main :: IO ()
main = do
    (trainData, testData) <- initMnist "./session9/data" 
    let trainMnist = V.MNIST {batchSize = 32, mnistData = trainData}
        testMnist  = V.MNIST {batchSize = 1,  mnistData = testData}

    putStrLn "Fixing 100 images for realtime validation..."
    fixedValidData <- forM [0 .. 99] $ \idx -> getItem testMnist idx

    initModel <- sample $ MLPSpec [784, 128, 64, 10] actFunc 

    putStrLn "Training Start"
    
    (trainedModel, totalValidLossHistory) <- foldLoop (initModel, []) epoch $ \(m, pastValidLoss) ep -> do
        
        (newM, epochValidLosses) <- runContT (streamFromMap (datasetOpts 2) trainMnist) $ trainLoop fixedValidData m GD . fst
        
        return (newM, pastValidLoss ++ epochValidLosses)

    drawLearningCurve "session9/loss_decrease_mnist.png" "Learning Curve" [ ("valid_loss", totalValidLossHistory) ]

    putStrLn ""
    putStrLn "Final evaluation on test dataset"
    evaluateModel trainedModel testMnist

    return ()