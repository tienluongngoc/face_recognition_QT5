from collections import OrderedDict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt 

class TRTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.engine = config["engine"]
        self.onnx = config["onnx"]
        device = config["device"]
        fp16 = config["fp16"]
        self.input_name = config["input_name"]
        self.output_name = config["output_name"]
        # self.export_engine()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(self.engine, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False 
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.context = model.create_execution_context()
        self.batch_size = bindings[self.input_name].shape[0]
        
        self.__dict__.update(locals()) 

    def forward(self, im):
        b, ch, h, w = im.shape
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        assert im.shape == self.bindings[self.input_name].shape, (im.shape, self.bindings[self.input_name].shape)
        
        self.binding_addrs[self.input_name] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings[self.output_name].data
        
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)

        return y
    
    def export_engine(self):
        half = False
        try:
            workspace = 4
            logger = trt.Logger(trt.Logger.INFO)
 
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            config.max_workspace_size = workspace * 1 << 30

            flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(self.onnx)):
                raise RuntimeError(f'failed to load ONNX file: {self.onnx}')

            if builder.platform_has_fast_fp16 and half:
                config.set_flag(trt.BuilderFlag.FP16)
            with builder.build_engine(network, config) as engine, open(self.engine, 'wb') as t:
                t.write(engine.serialize())
            return self.engine
        except Exception as e:
            print(e)

