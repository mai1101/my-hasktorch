--module MultipleRegression where

import Torch.Tensor (Tensor, asTensor, asValue, numel,reshape)
import Torch.Functional (add, mul, sub, sumAll, matmul, oneHot, stack, Dim(..))
import Torch (randIO')

import Control.Monad (forM_)

import ML.Exp.Chart (drawLearningCurve)


ys :: Tensor
ys = asTensor ([123, 290, 230, 261, 140, 173, 133, 179, 210, 181] :: [Float])
xs1 :: Tensor
xs1 = asTensor ([93, 230, 250, 260, 119, 183, 151, 192, 263, 185] :: [Float])
xs2 :: Tensor 
xs2 = asTensor ([150, 311, 182, 245, 152, 162, 99, 184, 115, 105] :: [Float])

xs :: Tensor
xs = stack (Dim 0) [xs1,xs2]


--勾配計算用
e :: Tensor
e = asTensor (0.0001 :: Float)

--学習率
learnRate :: Tensor
learnRate = asTensor (0.000052 :: Float)

--slope, intercept から input x に対応する estimatedY を求める関数
linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a1, a2] : 1 * 2 , b)
    Tensor ->           -- ^ data x : 2 * 10
    Tensor              -- ^ estimated y
linear (slope, intercept) input= add (matmul slope input) intercept

-- estimatedY と、真の値 yの差^2の合計/(2*要素数)を求める関数
cost ::
    Tensor -> -- ^ grand truth
    Tensor -> -- ^ estimated values
    Tensor    -- ^ loss: scalar
cost z z' = 
    let diff = sub z z'
    in (sumAll $ mul diff diff) / asTensor (fromIntegral (2 * (numel z)) :: Float)


--現在のaの値からそのaでのコスト関数の傾きを計算し、傾きと学習率から新しいaを求める関数
--多変量バージョン
calculateNewA :: 
    Tensor -> --a [1 * 2]
    Tensor -> --b
    Int ->    --multiple なので、aの中でupdateしたい index
    Tensor    --newA
calculateNewA a b index = 
    let mask = (oneHot 2 (asTensor index)) * e
        cost_y = cost ys (linear (a, b) xs)
        cost_y' = cost ys (linear (a+mask, b) xs)
        grad = (cost_y' - cost_y) / e
        updateTensor = (oneHot 2 (asTensor index)) * (learnRate * grad)
        newA = a - updateTensor
    in newA 


calculateNewB :: 
    Tensor -> --a [1 * 2]
    Tensor -> --b
    Tensor    --newB
calculateNewB a b = 
    let cost_y = cost ys (linear (a, b) xs)
        cost_y' = cost ys (linear (a, b+e) xs)
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
        newA = calculateNewA a b 0       --update first a
        newA' = calculateNewA newA b 1   --update second a
        newB = calculateNewB newA' b
    in trainLoop (epoch-1) (newA', newB) (currentCost : lst) 

    

main :: IO ()
main = do
    initialA <- randIO' [1,2]
    initialB <- randIO' [1]

    --before training
    let estimatedYs = linear (initialA, initialB) xs
        cost_y = cost ys estimatedYs

    --training
    let epoch = 100 :: Int
        (costList, (finalA, finalB)) = trainLoop epoch (initialA, initialB) []

    --after training
    let estimatedYs' = linear (finalA, finalB) xs
        cost_y' = cost ys estimatedYs'
        costList' = costList ++ [asValue cost_y']
      
    
    --print estimatedY and y 
    let estList = asValue (reshape [10] estimatedYs') :: [Float]
        ysList = asValue ys :: [Float]
    forM_ (zip ysList estList) $ \(y, estimatedY) -> do
        putStrLn $ "correct answer:" ++ show y
        putStrLn $ "estimated: " ++ show estimatedY
        putStrLn "******"
     

    putStrLn $ "cost before training :  " ++ show (asValue cost_y :: Float)
    putStrLn $ "cost  after training :  " ++ show (asValue cost_y' :: Float)

    putStrLn $ "finalA :  " ++ show (asValue finalA :: [[Float]])
    putStrLn $ "finalB :  " ++ show (asValue finalB :: Float)

    --learning curve
    drawLearningCurve "Session3/learning_curve3.png" "Learning Curve" [("cost", costList')]

