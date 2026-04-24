--module LinearRegression where

import Torch.Tensor (Tensor, asTensor, asValue, numel)
import Torch.Functional ( add, mul, sub, sumAll)
import Torch (randIO')

import Control.Monad (forM_)

import ML.Exp.Chart (drawLearningCurve)


ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])


--勾配計算用
e :: Tensor
e = asTensor (0.001 :: Float)

--学習率
learnRate :: Tensor
learnRate = asTensor (0.000056 :: Float)

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
    Tensor -> --a
    Tensor -> --b
    Tensor    --newA
calculateNewA a b = 
    let estimatedYs = linear (a, b) xs
        cost_y = cost ys estimatedYs
        estimatedYs' = linear (a+e, b) xs
        cost_y' = cost ys estimatedYs'
        grad = (cost_y' - cost_y) / e
        newA = a - learnRate * grad
    in newA 

calculateNewB :: 
    Tensor -> --a
    Tensor -> --b
    Tensor    --newB
calculateNewB a b = 
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
    (Tensor, Tensor) ->    --(a, b)  
    [Float] ->             --[cost]   
    ([Float], (Tensor, Tensor))       --(newA, newB)  
trainLoop 0 (a, b) lst = (reverse lst, (a, b))
trainLoop epoch (a, b) lst =
    let estimatedYs = linear (a, b) xs
        currentCost = asValue (cost ys estimatedYs) :: Float
        newA = calculateNewA a b
        newB = calculateNewB a b
    in trainLoop (epoch-1) (newA, newB) (currentCost : lst) 

    

main :: IO ()
main = do
    initialA <- randIO' [1]
    initialB <- randIO' [1]

    --before training
    let estimatedYs = linear (initialA, initialB) xs
        cost_y = cost ys estimatedYs

    --training
    let epoch = 700000 :: Int
        (costList, (finalA, finalB)) = trainLoop epoch (initialA, initialB) []

    --after training
    let estimatedYs' = linear (finalA, finalB) xs
        cost_y' = cost ys estimatedYs'
        costList' = costList ++ [asValue cost_y']
      
    --print estimatedY and y 
    let estList = asValue estimatedYs' :: [Float]
        ysList = asValue ys :: [Float]
    forM_ (zip ysList estList) $ \(y, estimatedY) -> do
        putStrLn $ "correct answer:" ++ show y
        putStrLn $ "estimated: " ++ show estimatedY
        putStrLn "******"
     

    putStrLn $ "cost before training :  " ++ show (asValue cost_y :: Float)
    putStrLn $ "cost  after training :  " ++ show (asValue cost_y' :: Float)

    putStrLn $ "finalA :  " ++ show (asValue finalA :: Float)
    putStrLn $ "finalB :  " ++ show (asValue finalB :: Float)

    --learning curve
    drawLearningCurve "Session3/learning_curve1.png" "Learning Curve" [("cost", costList')]

