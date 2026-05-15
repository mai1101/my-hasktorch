{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
module Admit where

import Control.Monad (when, forM_)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import ML.Exp.Chart (drawLearningCurve)  
import Data.Csv (decode, HasHeader(HasHeader), FromRecord)
import qualified Data.Vector as V
import qualified Data.ByteString.Lazy as BL
import qualified Evaluation as E

-- hyperparameter
numIters :: Int
numIters = 100     -- how many times to learn

learnRate :: Tensor
learnRate = 1e-2      -- learning rate

actFunc :: Tensor -> Tensor
actFunc = Torch.tanh   -- activation function


data Admission = Admission
    { gre       :: Float
    , toefl     :: Float
    , rating    :: Float
    , sop       :: Float
    , lor       :: Float
    , cgpa      :: Float
    , research  :: Float
    , chance    :: Float
    } deriving (Generic, Show, FromRecord)


-- read csvData and change it to list
loadData :: FilePath -> IO ([[Float]], [[Float]])
loadData path = do
    csvData <- BL.readFile path
    case decode HasHeader csvData :: Either String (V.Vector Admission) of
        Left err -> error err 
        Right vectorData -> do
                let listData = V.toList vectorData --vectorをlistへ
                let xsList = map (\d -> [ normalize (gre d) 290 340     
                        , normalize (toefl d) 92 120    
                        , normalize (rating d) 1 5                   
                        , normalize (sop d) 1 5
                        , normalize (lor d) 1 5
                        , normalize (cgpa d) 6.8 9.92                  
                        , research d                    
                        ]) listData
                let ysList = map (\d -> [chance d]) listData
                return (xsList, ysList)


-- narmalize : it takes value, min_value and max_value then passes normalized value
normalize :: Float -> Float -> Float -> Float
normalize x x_min x_max = (x - x_min) / (x_max - x_min)

-- splitData : it split data 8:1:1
splitData :: [[Float]] -> ([[Float]],[[Float]],[[Float]])
splitData xs = 
    let n = length xs
        n_train = (n * 8) `Prelude.div` 10
        n_valid = (n * 1) `Prelude.div` 10
        (train, rest) = splitAt n_train xs
        (valid, eval) = splitAt n_valid rest
    in (train, valid, eval)



--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

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


model :: MLP -> Tensor -> Tensor
model params t = mlp params t

main :: IO ()
main = do
    -- load data
    (xs, ys) <- loadData "Session5/data/AdmissionData.csv"
    let (train_xs_l, valid_xs_l, eval_xs_l) = splitData xs
    let (train_ys_l, valid_ys_l, eval_ys_l) = splitData ys

    -- change to tensor
    let train_xs = asTensor train_xs_l
        train_ys = asTensor train_ys_l
        valid_xs = asTensor valid_xs_l
        valid_ys = asTensor valid_ys_l
        eval_xs = asTensor eval_xs_l
        eval_ys = asTensor eval_ys_l

    -- initialize the model
    initModel <- sample $ MLPSpec [7,8,8,1] actFunc 

    --------------------------------------------------------------------------------
    -- train
    --------------------------------------------------------------------------------
    (trained, lossList) <- foldLoop (initModel, []) numIters $ \(state, pastLossList) i -> do

        --calculate loss
        let y  = train_ys                         -- y: true value
            y' = model state train_xs             -- y':estimated value
            loss = mseLoss y y'                   -- mse
            --loss = binaryCrossEntropyLoss' y y'  -- cross entropy version 

        --for validation
        let loss_valid = mseLoss valid_ys (model state valid_xs) -- mse
        --let loss_valid = binaryCrossEntropyLoss' valid_ys (model state valid_xs) -- cross entropy version
            currentLoss = asValue loss_valid :: Float   

        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss_valid  

        (newState, _) <- runStep state GD loss learnRate               
        return (newState, currentLoss : pastLossList)

    let finalLosses = reverse lossList
    drawLearningCurve "Session5/loss_decrease1.png" "Loss Curve" [("loss", finalLosses)]

    --------------------------------------------------------------------------------
    -- evaluate
    --------------------------------------------------------------------------------

    --  adjust trained model to evaluation data -> change tensor to list
    let finalY  = asValue (squeezeAll eval_ys) :: [Float]
        finalY' = asValue (squeezeAll (model trained eval_xs)) :: [Float]
 
    -- change float data to int data(0 or 1) by using threshold
    let threshold = 0.7
        y_true = map (\v -> if v >= threshold then 1 else 0) finalY
        y_pred = map (\v -> if v >= threshold then 1 else 0) finalY'

    putStrLn $ "Confusion Matrix  : " ++ show (E.makeConfusionMatrix 2 y_true y_pred)
    putStrLn $ "accurary          : " ++ show (E.accuracy y_true y_pred)
    putStrLn $ "macro-f1-Score    : " ++ show (E.macroF1Score 2 y_true y_pred)
    putStrLn $ "micro-f1-Score    : " ++ show (E.microF1Score 2 y_true y_pred)
    putStrLn $ "weighted-f1-Score : " ++ show (E.weightedF1Score 2 y_true y_pred)




    



    
