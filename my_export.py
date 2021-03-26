from config import  *
from crnn import FullCrnn,LiteCrnn,CRNNHandle
from PIL import Image
import numpy as np
import cv2
import torch
crnn_net =  LiteCrnn(32, 1, len(alphabet) + 1, nh, n_rnn=2, leakyRelu=False, lstmFlag=LSTMFLAG)

crnn_handle  =  CRNNHandle(crnn_model_path , crnn_net , gpu_id=0)

img = cv2.imread("testcrnn.bmp")

partImg = Image.fromarray(np.array(img))

partImg_ = partImg.convert('L')

simPred = crnn_handle.predict(partImg_)
output_onnx = 'crnn_lite_lstm_v2.onnx'
input_names = ["input"]
output_names = ["out"]
inputs = torch.randn(1, 1, 32, 277).to(self.device)
torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, input_names=input_names, output_names=output_names, keep_initializers_as_inputs=True, opset_version=11)