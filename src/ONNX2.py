# ONNX version (前処理変更)

import cv2
import time
import argparse
import onnxruntime
import numpy as np
from torchvision import transforms

def main(args):
     starttime = time.time()
     if args.gpu:
          session = onnxruntime.InferenceSession("./assets/model.onnx", \
               providers=['CUDAExecutionProvider'])
     else:
          session = onnxruntime.InferenceSession("./assets/model.onnx", \
               providers=['CPUExecutionProvider'])

     input_image = cv2.imread(args.filepath)
     input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

     preprocess = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize(256),  # (576, 324) -> (455, 256)
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])

     input_tensor = preprocess(input_image) 
     input_batch = input_tensor.unsqueeze(0) # (1, 3, 224, 224)
     input_batch = input_batch.to('cpu').detach().numpy().copy()

     input_name = session.get_inputs()[0].name
     output_name = session.get_outputs()[0].name

     outputs = session.run([output_name], {input_name: input_batch})[0]  # (1, 1000)
     final_outputs = outputs[0]    # array (1000)

     # top5
     toplabls = np.argsort(final_outputs)[::-1][:5]
     with open("./assets/imagenet_classes.txt", "r") as f:
          categories = [s.strip() for s in f.readlines()]

     probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

     for i in range(5):
          id = toplabls[i]
          print(categories[id], probs[id] * 100)

     endtime = time.time()   
     print(endtime - starttime)

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='ONNX version')
     parser.add_argument('--gpu', action='store_true', help='CPU or GPU')
     parser.add_argument('--filepath', default='./assets/test.png', help='file path')
     args = parser.parse_args()
     main(args)