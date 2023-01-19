# PyTorch version

import time
import copy
import torch
import argparse

from PIL import Image
from torchvision import transforms

def main(args):
     starttime = time.time()
     model = torch.load('./assets/model.pth')
     model.eval()

     input_image = Image.open(args.filepath)
     input_image = input_image.convert("RGB")

     preprocess = transforms.Compose([
          transforms.Resize(256),  # (576, 324) -> (455, 256)
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])

     input_tensor = preprocess(input_image) 
     input_batch = input_tensor.unsqueeze(0) # (1, 3, 224, 224)

     if torch.cuda.is_available() and args.gpu:
          if not args.fp16:
               model = model.to('cuda')
               input_batch = input_batch.to('cuda')
          else:
               model = model.to('cuda')
               model_fp16 = copy.deepcopy(model).half()
               input_batch_fp16 = input_batch.half()
               input_batch_fp16 = input_batch_fp16.to('cuda')
          
     with torch.no_grad():
          if args.gpu and args.fp16:
               output_fp16 = model_fp16(input_batch_fp16)
               output = output_fp16.float().cpu()
          else:
               output = model(input_batch)   # torch.size([1, 1000])

     probabilities = torch.nn.functional.softmax(output[0], dim=0)    # torch.size[1000]

     with open("./assets/imagenet_classes.txt", "r") as f:
          categories = [s.strip() for s in f.readlines()]
     top5_prob, top5_catid = torch.topk(probabilities, 5)
     for i in range(top5_prob.size(0)):
          print(categories[top5_catid[i]], top5_prob[i].item())

     endtime = time.time()   
     print(endtime - starttime)

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='PyTorch version')
     parser.add_argument('--gpu', action='store_true', help='CPU or GPU')
     parser.add_argument('--fp16', action='store_true', help='FP16')
     parser.add_argument('--filepath', default='./assets/test.png', help='file path')
     args = parser.parse_args()
     main(args)