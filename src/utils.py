import torch

def convert_onnx():
     """
     PyTorch model ---> ONNX
     """
     model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
     input = torch.randn(1, 3, 224, 224)
     torch.onnx.export(model,
                    input,                     # モデルの入力
                    "./assets/model.onnx",              # 保存ファイル名
                    export_params = True)      # 学習済みパラメータも保存

def convert_localmodel():
     model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
     torch.save(model, './assets/model.pth')

# convert_localmodel()