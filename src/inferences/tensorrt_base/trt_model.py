from collections import OrderedDict, namedtuple
from ..utils.yolov5_utils import *
import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt 
from utils.logger import Logger
import os.path
class TRTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(config.half)
        self.engine = config.engine
        self.onnx = config.onnx
        self.device = config.device
        self.half = config.half
        self.input_name = config.input_name
        self.output_name = config.output_name
        self.max_workspace_size = config.max_workspace
        self.logger = Logger.__call__().get_logger()
        
        # if not os.path.exists(selPyQt5

        self.logger.info(f'Loading {self.engine} for TensorRT inference...')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        trt_logger = trt.Logger(trt.Logger.INFO)
        
        with open(self.engine, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())

        context = model.create_execution_context()
        bindings = OrderedDict()
        self.fp16 = False
        self.dynamic = False
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            if model.binding_is_input(index):
                if -1 in tuple(model.get_binding_shape(index)):
                    self.dynamic = True
                    context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            shape = tuple(context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings[self.input_name].shape[0] 
        self.__dict__.update(locals()) 

    def forward(self, im):
        b, ch, h, w = im.shape
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        if self.dynamic and im.shape != self.bindings[self.input_name].shape:
            i_in, i_out = (self.model.get_binding_index(x) for x in (self.input_name, self.output_name))
            self.context.set_binding_shape(i_in, im.shape)
            self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=im.shape)
            self.bindings[self.output_name].data.resize_(tuple(self.context.get_binding_shape(i_out)))
        s = self.bindings[self.input_name].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs[self.input_name] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings[self.output_name].data
        return y

    
    def export_engine(self):
        try:
            self.logger.info(f'TensorRT starting export with TensorRT {trt.__version__}...')
            trt_logger = trt.Logger(trt.Logger.INFO)

            builder = trt.Builder(trt_logger)
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size * 1 << 30

            flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, trt_logger)
            if not parser.parse_from_file(str(self.onnx)):
                raise RuntimeError(f'failed to load ONNX file: {self.onnx}')

            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            outputs = [network.get_output(i) for i in range(network.num_outputs)]
            self.logger.info(f'TensorRT Network Description:')
            for inp in inputs:
                self.logger.info(f'TensorRT input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
            for out in outputs:
                self.logger.info(f'TensorRT output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

            self.logger.info(f'TensorRT building FP{16 if builder.platform_has_fast_fp16 and self.half else 32} engine in {self.engine}')

            if builder.platform_has_fast_fp16 and self.half:
                config.set_flag(trt.BuilderFlag.FP16)
            with builder.build_engine(network, config) as engine, open(self.engine, 'wb') as t:
                t.write(engine.serialize())
            self.logger.info(f'TensorRT export success, saved as {self.engine} ({file_size(self.engine):.1f} MB)')
            return self.engine
        except Exception as e:
            self.logger.info("TensorRT", e)

