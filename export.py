import torch
from torch.autograd import Variable
import utils
import lib.models.crnn as crnn
import struct

model_path = './weights/mycrnn.pth'

# model = crnn.CRNN(32, 1, 37, 256)
# image = torch.ones(1, 1, 32, 100)
#
model = crnn.CRNN(32, 1, 6736, 256)
image = torch.ones(1, 1, 32, 492)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
checkpoint = torch.load(model_path)
if 'state_dict' in checkpoint.keys():
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
# model.load_state_dict(torch.load(model_path))


if torch.cuda.is_available():
    image = image.cuda()
input_bame=['input']
out_name=['out']
onnx_f = "mycrnn.onnx"
torch.onnx.export(model, image, onnx_f,input_names=input_bame,output_names=out_name,verbose=False, opset_version=9,export_params=True,keep_initializers_as_inputs=True)
model.eval()
print(model)
print('image shape ', image.shape)
preds = model(image)

f = open("mycrnn.wts", 'w')
f.write("{}\n".format(len(model.state_dict().keys())))
for k,v in model.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")

