module Evaluation where

import Data.List

-- accuracy function
-- Number of correct / Total number of elements
accuracy :: [Int] -> [Int] -> Float
accuracy y_true y_pred = 
    let correct = length $ filter (==True) $ zipWith (==) y_true y_pred
        len = length y_true
    in (fromIntegral correct) / (fromIntegral len) 

-- precision function
-- The proportion of cases that are actually 'A' out of those predicted to be 'A'
precision :: Int -> [Int] -> [Int] -> Float
precision target y_true y_pred = 
    let y_pair = zip y_true y_pred
        predictTarget = filter (\(t,p) -> p == target) y_pair
        truePositive = filter (\(t,p) -> t==target) predictTarget
        len_tp = length truePositive
        len_target = length predictTarget
    in (fromIntegral len_tp) / (fromIntegral len_target) 

-- recall function
-- The proportion of cases that are actually 'A' among those predicted to be 'A'
recall :: Int -> [Int] -> [Int] -> Float
recall target y_true y_pred = 
    let y_pair = zip y_true y_pred
        haveTarget = filter (\(t,p) -> t == target) y_pair
        truePositive = filter (\(t,p) -> p==target) haveTarget
        len_tp = length truePositive
        len_target = length haveTarget
    in (fromIntegral len_tp) / (fromIntegral len_target) 


-- n: numbers of classes
-- Row    : Actual value
-- Column : Predicted value
makeConfusionMatrix :: Int -> [Int] -> [Int] -> [[Int]]
makeConfusionMatrix n y_true y_pred = 
    let classes = [0..(n-1)]
        y_pair = zip y_true y_pred
    in [ [ countPair i j y_pair | j <- classes ] | i <- classes ]

countPair :: Int -> Int -> [(Int,Int)] -> Int
countPair i j y_pair = 
    let true_i_pred_j = filter  (\(t,p) -> p == j && t == i) y_pair
    in length true_i_pred_j

-- F1 score for a given label
calcF1Score :: Int -> [Int] -> [Int] -> Float
calcF1Score target y_true y_pred = 
    let p = precision target y_true y_pred 
        r = recall target y_true y_pred 
    in (2*p*r) / (p+r)


macroF1Score :: Int -> [Int] -> [Int] -> Float
macroF1Score n y_true y_pred = 
    let classes = [0..(n-1)]
        f1Scores = [calcF1Score m y_true y_pred | m <- classes]
    in (sum f1Scores) / (fromIntegral n) 

--　Weighted F1 score
weightedF1Score :: Int -> [Int] -> [Int] -> Float
weightedF1Score n y_true y_pred = 
    let classes = [0..(n-1)]
        f1Scores = [calcF1Score m y_true y_pred | m <- classes]
        supports = [fromIntegral $ countSupport t y_true | t <- classes]
        totalLen = fromIntegral $ length y_true 
        weights = map (\s -> s / totalLen) supports
    in sum $ zipWith (*) f1Scores weights

-- Count the number of occurrences of “target”
countSupport :: Int -> [Int] -> Int
countSupport target y_true = length $ filter  (\t -> t==target) y_true


microF1Score :: Int -> [Int] -> [Int] -> Float
microF1Score n y_true y_pred = 
    let classes = [0..(n-1)]
        y_pair = zip y_true y_pred
        total_TP = fromIntegral $ sum [countTP m y_pair | m <- classes]
        total_FP = fromIntegral $ sum [countFP m y_pair | m <- classes]
        total_FN = fromIntegral $ sum [countFN m y_pair | m <- classes]
    in total_TP / (total_TP + 0.5 * (total_FP + total_FN))

countTP :: Int -> [(Int,Int)] -> Int
countTP target y_pair = length $ filter (\(t,p) -> t==target && p==target) y_pair 

countFP :: Int -> [(Int,Int)] -> Int
countFP target y_pair = length $ filter (\(t,p) -> t/=target && p==target) y_pair

countFN :: Int -> [(Int,Int)] -> Int
countFN target y_pair = length $ filter (\(t,p) -> t==target && p/=target) y_pair


