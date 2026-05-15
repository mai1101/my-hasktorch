{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
module Titanic where

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
numIters = 8000     -- how many times to learn

learnRate :: Tensor
learnRate = 0.08     -- learning rate

actFunc :: Tensor -> Tensor
actFunc = Torch.tanh   -- activation function

data Titanic = Titanic
    { survived :: Float -- Dependent variable
    , pclass   :: Float
    , sex      :: Float -- male=0, female=1 
    , age      :: Float -- null=29.7
    , sibSp    :: Float
    , parch    :: Float
    , fare     :: Float
    , embarked :: Float -- S=0, C=1, Q=2, (null=0)
    } deriving (Generic, Show, FromRecord)

data RawTitanic = RawTitanic
    { r_passengerId :: String
    , r_survived    :: String
    , r_pclass      :: String
    , r_name        :: String
    , r_sex         :: String
    , r_age         :: String
    , r_sibSp       :: String
    , r_parch       :: String
    , r_ticket      :: String
    , r_fare        :: String
    , r_cabin       :: String
    , r_embarked    :: String
    } deriving (Generic, FromRecord)


-- change raw titanic data to usable format
cleanData :: [RawTitanic] -> [Titanic]
cleanData raws = map convert raws
  where
    meanAge = 29.7 --if age is null, input 30
    convert r = Titanic 
        { survived = read (r_survived r)
        , pclass = read (r_pclass r)
        , sex = if r_sex r == "male" then 0.0 else 1.0
        , age = if null (r_age r) then meanAge else read (r_age r)
        , sibSp = read (r_sibSp r)
        , parch = read (r_parch r)
        , fare = read (r_fare r)
        , embarked = case r_embarked r of -- 文字列を数値に変換
                       "S" -> 0.0
                       "C" -> 1.0
                       "Q" -> 2.0
                       _   -> 0.0
        }

-- read csv data and change it to list 
loadData :: FilePath -> IO ([[Float]], [[Float]])
loadData path = do
    csvData <- BL.readFile path
    case decode HasHeader csvData :: Either String (V.Vector RawTitanic) of
        Left err -> error err 
        Right vectorData -> do
                --let cleanedData = cleanData vectorData
                let listData' = V.toList vectorData  --vectorをlistへ
                let listData = cleanData listData'
                let xsList = map (\d -> 
                        [ normalize (pclass d) 1 3
                        , sex d           
                        , normalize (age d) 0.42 80
                        , normalize (sibSp d) 0 8
                        , normalize (parch d) 0 6           
                        , normalize (fare d) 0 512   
                        , normalize (embarked d) 0 2            
                        ]) listData
                let ysList = map (\d -> [survived d]) listData
                return (xsList, ysList)

normalize :: Float -> Float -> Float -> Float
normalize x x_min x_max = (x - x_min) / (x_max - x_min)

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
  { feature_counts :: [Int],             -- neutron list
    nonlinearitySpec :: Tensor -> Tensor -- activation function
  }

data MLP = MLP
  { layers :: [Linear],                  -- weight and bias list
    nonlinearity :: Tensor -> Tensor     -- activation function
  }
  deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where -- initialize layer by random value
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
    -- load data and split it 8:1:1  
    (xs, ys) <- loadData "Session5/data/titanic/train.csv"
    let (train_xs_l, valid_xs_l, eval_xs_l) = splitData xs
    let (train_ys_l, valid_ys_l, eval_ys_l) = splitData ys

    -- change list to tensor
    let train_xs = asTensor train_xs_l
        train_ys = asTensor train_ys_l
        valid_xs = asTensor valid_xs_l
        valid_ys = asTensor valid_ys_l
        eval_xs = asTensor eval_xs_l
        eval_ys = asTensor eval_ys_l

    -- initialize the model
    initModel <- sample $ MLPSpec [7,16,16,1] actFunc 
    
    --------------------------------------------------------------------------------
    -- train
    --------------------------------------------------------------------------------
    (trained, lossList) <- foldLoop (initModel, []) numIters $ \(state, pastLossList) i -> do
        --calculate loss
        let y  = train_ys                         -- y: correct value
            y' = model state train_xs             -- y': estimated value
            loss = mseLoss y y'                   -- y,y'の平均二乗誤差

        --for validation
        let loss_valid = mseLoss valid_ys (model state valid_xs)  
            currentLoss = asValue loss_valid :: Float   

        when (i `mod` 1000 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss_valid  

        (newState, _) <- runStep state GD loss learnRate                  
        return (newState, currentLoss : pastLossList)

    let finalLosses = reverse lossList
    drawLearningCurve "Session5/loss_decrease2.png" "Loss Curve" [("loss", finalLosses)]

    --------------------------------------------------------------------------------
    -- evaluate
    --------------------------------------------------------------------------------
    let finalY  = asValue (squeezeAll eval_ys) :: [Float]
        finalY' = asValue (squeezeAll (model trained eval_xs)) :: [Float]

    let y_true = map Prelude.floor finalY
        y_pred = map (\v -> if v >= 0.5 then 1 else 0) finalY'

    putStrLn $ "Confusion Matrix  : " ++ show (E.makeConfusionMatrix 2 y_true y_pred)
    putStrLn $ "accurary          : " ++ show (E.accuracy y_true y_pred)
    putStrLn $ "macro-f1-Score    : " ++ show (E.macroF1Score 2 y_true y_pred)
    putStrLn $ "micro-f1-Score    : " ++ show (E.microF1Score 2 y_true y_pred)
    putStrLn $ "weighted-f1-Score : " ++ show (E.weightedF1Score 2 y_true y_pred)


    



    
