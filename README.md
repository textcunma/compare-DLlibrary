# compare-DLlibrary
深層学習ライブラリのCPU推論速度比較

## 概要
画像認識タスクにおいて、PyTorch, OpenCV, ONNXの推論速度比較を行います。
使用モデルは「densenet121」とします。

その他、以下は省きます
- FP16 (PyTorch - GPU推論のみ)
- TensorRT (GPU推論のみ)
- LibTorch
- OpenCV (GPU推論)
- OpenVINO

## 実行環境
|  ライブラリ名  |  バージョン  |
| ---- | ---- |
|  numpy  |  1.24.1  |
|  torch  |  1.12.1+cu113  |
|  torchvision  |  0.13.1+cu113  |
|  Pillow  |  9.4.0  |
|  opencv-python  |  4.7.0.68  |
|  onnxruntime  |  1.13.1  |

## 環境構築手順
1. venvによって環境を構築
     ```
     python -m venv compareEnv
     ```

2. 環境を有効化
     ```
     // Windows
     .\compareEnv\Scripts\activate

     // Linux
     source compareEnv/bin/activate
     ```

3. ライブラリをダウンロード
     ```
     pip install -r requirements.txt
     ```
     
## 実験
3回実行してその平均実行時間を結果としました。

|  ライブラリ名  |  平均実行時間[s]  |　推定ラベル名 | 信頼度 |
| ---- | ---- | ---- | ---- | 
|  PyTorch  |  0.18(0.178)  | steel arch bridge | 0.84 |
|  ONNX  |  0.14(0.142)  | steel arch bridge | 0.88 |
|  ONNX (前処理変更)  |  0.14(0.144)  | steel arch bridge | 0.84 |
|  OpenCV  | 0.14(0.139) | steel arch bridge | 0.88 |