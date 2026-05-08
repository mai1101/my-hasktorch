{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module MlpXor where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import Control.Monad (forM_)  

-- hyperparameter
batchSize :: Int
batchSize = 2           -- 一度の学習で計算するデータ数

numIters :: Int
numIters = 2000         -- 何回学習するか

learnRate :: Tensor
learnRate = 1e-1        -- 学習率

actFunc :: Tensor -> Tensor
actFunc = Torch.tanh    -- 活性化関数

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],             -- 各層のニューロンの数のリスト 
    nonlinearitySpec :: Tensor -> Tensor -- 活性化関数
  }

data MLP = MLP
  { layers :: [Linear],                  -- 各層における重みとバイアス　のリスト　
    nonlinearity :: Tensor -> Tensor     -- 活性化関数
  }
  deriving (Generic, Parameterized)

--LinearSpec: 一つの層の入力数、出力数を受け取って、一つの層のスペック（仕様書）を返す
--sample : 層のスペックを受け取って、 ランダムな初期値を持たせたLinear型 (=層ごとの重みとバイアスのデータ)を返す
instance Randomizable MLPSpec MLP where --各層のランダムな初期化
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts --隣り合う層のサイズをペアにして、数珠つなぎのリストを作る [(入力数、出力数)]
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) = scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

--map linear layers [wx1+b, wx2+b ... ]　関数のリスト
--intersperse nonlinearity 活性化関数の適用を層の計算の間に入れる
mlp :: MLP -> Tensor -> Tensor 
mlp MLP {..} input = foldl' (\x f -> f x) input $ intersperse nonlinearity $ map linear layers 

-- XOR計算をする関数
tensorXOR :: Tensor -> Tensor
tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
  where
    a = select 1 0 t
    b = select 1 1 t

-- ステップ関数: 0以上なら1.0, 0未満なら0.0を返す
stepFunc :: Tensor -> Tensor
stepFunc x = toType Float $ ge x (asTensor (0.0 :: Float))

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

model :: MLP -> Tensor -> Tensor
model params t = mlp params t

main :: IO ()
main = do
  -- ランダムな重みとバイアスでモデルを初期化
  -- 入力層2, 隠れ層2, 出力層1
  initModel <- sample $ MLPSpec [2, 2, 1] actFunc    

  trained <- foldLoop initModel numIters $ \state i -> do
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5) --XORの入力データをランダムに2セット生成 ex. [[0,1],[1,1]]
    let y  = tensorXOR input                  -- y:正しい値 
        y' = squeezeAll $ model state input   -- y':予測値
        loss = mseLoss y y'                   -- y,y'の平均二乗誤差

    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss  -- 誤差を表示

    (newState, _) <- runStep state GD loss learnRate                  -- 重みを更新し、新しいモデルをnewStateに格納
    return newState

  putStrLn "Final Model:"
  forM_ ([[0,0],[0,1],[1,0],[1,1]::[Float]]) $ \x -> do
    putStr $ show x ++ " => "
    putStrLn $ show $ squeezeAll  $ model trained $ asTensor x 