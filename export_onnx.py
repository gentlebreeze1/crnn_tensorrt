import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='weights/checkpoint_6_acc_0.9764.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    print('raw img shape: hxw={}x{}'.format(h, w))
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    
    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    print('resied to 32,x img shape: hxw={}x{}'.format(h, w))
    
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))
    return img


if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    started = time.time()

    # img_raw = cv2.imread(args.image_path)
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    #
    # in_im = recognition(config, img, model, converter, device)
    # print('input image shape: ', in_im.shape)
    # finished = time.time()
    #print('elapsed time: {0}'.format(finished - started))
    input_bame=['input']
    out_name=['out']
    input = Variable(torch.randn(1, 1, 32, 492)).cuda()
    onnx_f = args.checkpoint.replace('.pth', '.onnx')
    torch.onnx.export(model, input, onnx_f,input_names=input_bame,output_names=out_name,verbose=False, opset_version=11,export_params=True,keep_initializers_as_inputs=True)

    # cv2.imshow('raw', img_raw)
    # cv2.waitKey(0)

    

