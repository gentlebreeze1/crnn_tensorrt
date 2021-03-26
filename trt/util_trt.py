# tensorrt-lib

import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from trt.calibrator import Calibrator
from trt.common import *
from trt.convert_trt_quant import preprocess_v1
from torch.autograd import Variable
import torch
import numpy as np
import time
import cv2
# add verbose
out_shapes=[()]
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # ** engine可视化 **d
def detect(image_path,engine_path):
    image=cv2.imread(image_path)
    image=preprocess_v1(image)
    with get_engine1(engine_path=engine_path) as engine,engine.create_execution_context() as context:
        inputs,outputs,bindings,stream=allocate_buffers(engine)
        inputs[0].host=image
        trt_outputs=do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
    trt_outputs=[output.reshape(shape) for output,shape in zip(trt_outputs,out_shapes)]
    return trt_outputs
def get_engine1(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# create tensorrt-engine
  # fixed and dynamic
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
               fp16_mode=False, int8_mode=False, calibration_stream=None, calibration_table_path="", save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1) as network,\
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
                assert network.num_layers > 0, 'Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible '
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))        
            
            # build trt engine
            builder.max_batch_size = max_batch_size
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.fp16_mode = fp16_mode
            if int8_mode:
                builder.int8_mode = int8_mode
                assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                builder.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
                print('Int8 mode enabled')
            engine = builder.build_cuda_engine(network) 
            if engine is None:
                print('Failed to create the engine')
                return None   
            print("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)
