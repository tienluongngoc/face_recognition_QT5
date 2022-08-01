import torch.nn as nn
import torch
import tensorrt_test as trt 
import logging
from collections import OrderedDict, namedtuple
import numpy as np
import tensorrt_test as trt 

# from utils.logger import Logger
# logger = Logger.__call__().get_logger()
# logger.info("Face recognition, initialized")


# LOGGER = logging.getLogger("yolov5")

# class RetinaFace(nn.Module):
#     def __init__(self, weight="models/R50.engine", device = torch.device('cpu')) -> None:
#         super().__init__()

#         logger = trt.Logger(trt.Logger.INFO)
#         with open(weight, 'rb') as f, trt.Runtime(logger) as runtime:
#             model = runtime.deserialize_cuda_engine(f.read())

#         Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

#         self.bindings = OrderedDict()
#         fp16 = False  # default updated below
#         for index in range(model.num_bindings):
#             name = model.get_binding_name(index)
#             dtype = trt.nptype(model.get_binding_dtype(index))
#             shape = tuple(model.get_binding_shape(index))
#             data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
#             self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
#             if model.binding_is_input(index) and dtype == np.float16:
#                 fp16 = True
#         self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
#         self.context = model.create_execution_context()
#         self.batch_size = self.bindings['data'].shape[0]
        
#         # self.__dict__.update(locals())  # assign all variables to self

#     def forward(self, img):
#         print(img.data_ptr())
#         self.binding_addrs['data'] = int(img.data_ptr())
#         self.context.execute_v2(list(self.binding_addrs.values()))
#         y = self.bindings['output'].data

#         if isinstance(y, np.ndarray):
#             y = torch.tensor(y, device=self.device)

#         return y

#     def image_preporcessing(self, img):
#         scales = [640, 640]
#         im_shape = img.shape
#         target_size = scales[0]
#         max_size = scales[1]
#         im_size_min = np.min(im_shape[0:2])
#         im_size_max = np.max(im_shape[0:2])
#         im_scale = float(target_size) / float(im_size_min)
#         if np.round(im_scale * im_size_max) > max_size:
#             im_scale = float(max_size) / float(im_size_max)

#         print('im_scale', im_scale)

#         scales = [im_scale]
#         flip = False


#     def warmup(self, imgsz = (1,3,640,640)):
#         pass


class YOLOV5TRT(nn.Module):
    def __init__(self, weights='models/R50.engine', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        super().__init__()
        stride, names = 32, [f'class{i}' for i in range(1000)]  # assign defaults
        pt = False
        # if data:  # assign class names (optional)
        #     with open(data, errors='ignore') as f:
        #         names = yaml.safe_load(f)['names']
        
        # LOGGER.info(f'Loading {weights} for TensorRT inference...'
        # check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False  # default updated below
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
        self.batch_size = bindings['data'].shape[0]
        
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        print(im.shape)
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16


        # print((im.shape, self.bindings['data'].shape))
        assert im.shape == self.bindings['data'].shape, (im.shape, self.bindings['data'].shape)
        
        self.binding_addrs['data'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data
        
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)

        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type != 'cpu':
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1): 
                self.forward(im) 