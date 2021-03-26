import torch
from torch.autograd import Variable
import tensorrt as trt
from trt.common import  *
import numpy as np
import cv2
from lib.utils import utils as utils
from lib.config import alphabets as alphabets
engine_file="/data/zhangyong/shenli/crnn_cn_pt-master/trt/model/mycrnn7.trt"
image_path="/data/zhangyong/shenli/crnn_cn_pt-master/images/test.png"
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
def get_engine1(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
def main():
    img_raw = cv2.imread(image_path)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(alphabets.alphabet)
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=32 / h, fy=32 / h,
                     interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (280 / 160))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (32, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - 0.588) / 0.193
    img = img.transpose([2, 0, 1])
    with get_engine1(engine_file) as engine,engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = img
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    output=trt_outputs[0]
    output=output.reshape((124,1,-1))
    output=torch.from_numpy(output)
    _, preds = output.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

if __name__=="__main__":
    main()