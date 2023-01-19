# OpenCV version

import cv2
import time
import argparse
import numpy as np

def main(args):
     starttime = time.time()
     model = cv2.dnn.readNetFromONNX('./assets/model.onnx')   # ONNXモデル読み込み
     input_image = cv2.imread(args.filepath)

     # Resize(256)  (h, w) = (324, 576) ==> (256, 455)
     h, w, _ = input_image.shape
     if h > w:
          input_image = cv2.resize(input_image, (int(256 * h / w) , 256))
     else:
          input_image = cv2.resize(input_image, (int(256 * w / h), 256))

     # CenterCrop(224)
     h, w, _ = input_image.shape   # h = 256, w = 144
     center_x = w // 2   # 72
     center_y = h // 2   # 128
     input_image = input_image[center_y - 112 : center_y + 112, 
                                   center_x - 112 : center_x + 112]

     # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     input_image = input_image.astype('float32') / 255.0
     input_image -= np.array([0.485, 0.456, 0.406])
     input_image /= np.array([0.229, 0.224, 0.225])
     
     blob = cv2.dnn.blobFromImage(image = input_image, swapRB = True) # (1, 3, 224, 224)

     model.setPreferableBackend( cv2.dnn.DNN_BACKEND_OPENCV )
     model.setPreferableTarget( cv2.dnn.DNN_TARGET_CPU )

     model.setInput(blob)
     outputs = model.forward()     # array (1, 1000)
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
     parser = argparse.ArgumentParser(description='OpenCV version')
     parser.add_argument('--filepath', default='./assets/test.png', help='file path')
     args = parser.parse_args()
     main(args)