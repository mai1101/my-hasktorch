module PerceptronAndGate where

import Torch.Tensor (Tensor, asTensor,asValue, reshape)
import Torch.Functional (add, mul, sub, sumAll, matmul, ge, transpose2D)
import Torch (randIO')
import Torch (DType(..), toType)
import ML.Exp.Chart (drawLearningCurve)

-- learning rate
learnRate :: Tensor
learnRate = asTensor (0.01 :: Float)

-- epoch
epoch :: Int
epoch = 50

trainingData :: [([Float],Float)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

-- step function : if argument is greater than 0 then return 0 else 1
step :: Tensor -> Tensor
step x = toType Float $ ge x 0

-- x1*w1 + x2w2 + b 
perceptron ::
    Tensor ->  -- x
    Tensor ->  -- weights
    Tensor ->  -- bias
    Tensor     -- output
perceptron xs ws b = add b $ matmul xs ws 

-- calculate error
calculateError ::
    Tensor ->   -- true value
    Tensor ->   -- output value
    Tensor      -- error
calculateError trueValue z = sub trueValue z

-- update weight
calculateNewW ::
    Tensor ->   -- w 
    Tensor ->   -- error
    Tensor ->   -- x
    Tensor      -- NewW
calculateNewW w e x = 
    let gradW = matmul (transpose2D x) e
    in add w (mul learnRate gradW)

-- update bias
calculateNewB ::
    Tensor ->   -- b 
    Tensor ->   -- error
    Tensor      -- NewB
calculateNewB b e = 
    let gradB = sumAll e
    in add b (mul learnRate gradB)

-- train
trainLoop :: 
    Int ->                 --epoch
    Tensor ->              --input
    Tensor ->              --true output
    (Tensor, Tensor) ->    --(w, b)    
    ((Tensor, Tensor), [Float])      --(newW, newB, [loss])  
trainLoop 0 _ _ (w, b) = ((w, b), [])
trainLoop count x trueValue (w, b) = 
    let output = step $ perceptron x w b
        er = calculateError trueValue output
        currentLoss = asValue (sumAll (mul er er)) :: Float
        newW = calculateNewW w er x
        newB = calculateNewB b er
        ((finalW, finalB), restLoss) = trainLoop (count-1) x trueValue (newW, newB)
    in ((finalW, finalB), (currentLoss : restLoss)) -- add currentLoss to list of loss





main :: IO ()
main = do
    -- initialize parameters(weights and bias) with random values.
    w <- randIO' [2,1] --weight
    b <- randIO' [1]  --bias

    let x = asTensor (map fst trainingData) -- input
        trueValue = reshape [4, 1] $ asTensor (map snd trainingData) -- true output = Tensor[1,0,0,0]
 
    -- train
    let ((finalW,finalB),lossList) = trainLoop epoch x trueValue (w, b)
        p = perceptron x finalW finalB
        output = step p

    putStrLn $ "Prediction: " ++ show (asValue output ::[Float])
    putStrLn $ "Final Weights: " ++ show (asValue finalW :: [Float])
    putStrLn $ "Final Bias: " ++ show (asValue finalB :: Float)

    drawLearningCurve "Session4/loss_decrease1.png" "Loss Curve" [("loss", lossList)]


