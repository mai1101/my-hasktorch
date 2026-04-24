--module GraduateAdmissionLinear where

import Torch.Tensor (Tensor, asTensor, asValue, numel)
import Torch.Functional (add, mul, sub, sumAll)
import Torch (randIO')

import ML.Exp.Chart (drawLearningCurve)

import Data.Csv (decode, HasHeader(HasHeader))
import qualified Data.Vector as V
import qualified Data.ByteString.Lazy as BL


--勾配計算用
e :: Tensor
e = asTensor (0.0001 :: Float)

--学習率
learnRate :: Tensor
learnRate = asTensor (0.0263 :: Float)

--2行の csv data を読み込み、tensor に変換
loadData :: FilePath -> IO (Tensor, Tensor)
loadData path = do
    csvData <- BL.readFile path
    case decode HasHeader csvData :: Either String (V.Vector (Float, Float)) of
        Left err -> error err 
        Right vectorData -> do
                let listData = V.toList vectorData --vectorをlistへ
                let (xsList, ysList) = unzip listData
                return (asTensor xsList, asTensor ysList)

--slope, intercept から input x に対応する estimatedY を求める関数
linear :: 
    (Tensor, Tensor) -> -- ^ parameters (a, b)
    Tensor ->           -- ^ data x
    Tensor              -- ^ estimated y
linear (slope, intercept) input = add (mul slope input) intercept

-- estimatedY と、真の値 yの差^2の合計/(2*要素数)を求める関数
cost ::
    Tensor -> -- ^ grand truth
    Tensor -> -- ^ estimated values
    Tensor    -- ^ loss: scalar
cost z z' = 
    let diff = sub z z'
    in (sumAll $ mul diff diff) / asTensor (fromIntegral (2 * (numel z)) :: Float)


--現在のaの値からそのaでのコスト関数の傾きを計算し、傾きと学習率から新しいaを求める関数
--傾きは = (aでのcost) - (a+eでのcost) / e      eは微小な値
calculateNewA :: 
    Tensor -> --xs
    Tensor -> --ys
    Tensor -> --a
    Tensor -> --b
    Tensor    --newA
calculateNewA xs ys a b = 
    let estimatedYs = linear (a, b) xs
        cost_y = cost ys estimatedYs
        estimatedYs' = linear (a+e, b) xs
        cost_y' = cost ys estimatedYs'
        grad = (cost_y' - cost_y) / e
        newA = a - learnRate * grad
    in newA 

calculateNewB :: 
    Tensor -> --xs
    Tensor -> --ys
    Tensor -> --a
    Tensor -> --b
    Tensor    --newB
calculateNewB xs ys a b = 
    let estimatedYs = linear (a, b) xs
        cost_y = cost ys estimatedYs
        estimatedYs' = linear (a, b+e) xs
        cost_y' = cost ys estimatedYs'
        grad = (cost_y' - cost_y) / e
        newB = b - learnRate * grad 
    in newB 


--train 残りの繰り返し回数と、(a,b)を受け取って、更新した(a,b)を返す関数
trainLoop :: 
    Int ->                 --epoch   
    (Tensor, Tensor) ->           --(train_xs, train_ys)
    (Tensor, Tensor) ->           --(valid_xs, valid_ys)
    (Tensor, Tensor) ->    --(a, b)  
    [Float] ->             --[train_cost]
    [Float] ->             --[valid_cost]   
    ([Float],[Float], (Tensor, Tensor))       --([train_cost],[valid_cost],(newA, newB))  
trainLoop 0 _ _ (a, b) t_cost v_cost = (reverse t_cost, reverse v_cost, (a, b))
trainLoop epoch (t_xs, t_ys) (v_xs, v_ys) (a, b) t_cost v_cost = 
    let t_estimatedYs = linear (a, b) t_xs
        v_estimatedYs = linear (a, b) v_xs
        t_curCost = asValue (cost t_ys t_estimatedYs) :: Float
        v_curCost = asValue (cost v_ys v_estimatedYs) :: Float
        newA = calculateNewA t_xs t_ys a b
        newB = calculateNewB t_xs t_ys a b
    in trainLoop (epoch-1) (t_xs, t_ys) (v_xs, v_ys) (newA, newB) (t_curCost : t_cost) (v_curCost : v_cost)



main :: IO ()
main = do
    initialA <- randIO' [1]
    initialB <- randIO' [1]

    (train_xs, train_ys) <- loadData "Session3/data/train.csv"
    (valid_xs, valid_ys) <- loadData "Session3/data/valid.csv"

    --training
    let epoch = 30000 :: Int 
        (train_cost, valid_cost, (finalA, finalB)) = trainLoop epoch (train_xs, train_ys) (valid_xs, valid_ys) (initialA, initialB) [] []

    putStrLn $ "finalA :  " ++ show (asValue finalA :: Float)
    putStrLn $ "finalB :  " ++ show (asValue finalB :: Float)

    --evaluate
    (eval_xs, eval_ys) <- loadData "Session3/data/eval.csv"
    let evalPredYs = linear (finalA, finalB) eval_xs
        evalCost = cost eval_ys evalPredYs 

    --print evaluate cost
    putStrLn $ "evaluate cost :  " ++ show (asValue evalCost :: Float)

    --learning curve
    drawLearningCurve "Session3/learning_curve2.png" "Learning Curve" [("train", train_cost),("valid", valid_cost)]