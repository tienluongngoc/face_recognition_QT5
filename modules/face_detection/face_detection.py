
import yaml
from modules.utils.tensorrt_model import TensorrtInference
from modules.utils.detection import *
class FaceDetection(TensorrtInference):
    def __init__(self, config_fp) -> None:
        with open(config_fp) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.onnx = config["face_detection"]["onnx"]
        self.engine = config["face_detection"]["engine"]
        self.fp16_mode = config["face_detection"]["fp16_mode"]
        self.output_shape = config["face_detection"]["output_shape"]
        self.input_shape = config["face_detection"]["input_shape"]
        self.stride_max = config["face_detection"]["stride_max"]
        self.confidence_threshold = config["face_detection"]["confidence_threshold"]
        self.iou_threshold = config["face_detection"]["iou_threshold"]

        super().__init__(self.onnx, self.engine, self.fp16_mode)

    def detect(self, img):
        img = img.numpy()
        pred = self(img)
        return pred.reshape(self.output_shape) 
    
    def pre_process(self, orgimg):
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]
        r = self.input_shape[0]/ max(h0, w0) 
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        imgsz = check_img_size(self.input_shape[0], s=self.stride_max) 
        img = letterbox(img0, new_shape=imgsz,auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img,orgimg

    def post_process(self, prediction, classes=None, agnostic=False, labels=()):
        prediction = torch.from_numpy(prediction)
        nc = prediction.shape[2] - 15 
        xc = prediction[..., 4] > self.confidence_threshold 
        min_wh, max_wh = 2, 4096
        time_limit = 10.0 
        redundant = True
        multi_label = nc > 1 
        merge = False 

        t = time.time()
        output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction): 
            x = x[xc[xi]]
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 15), device=x.device)
                v[:, :4] = l[:, 1:5]
                v[:, 4] = 1.0  
                v[range(len(l)), l[:, 0].long() + 15] = 1.0 
                x = torch.cat((x, v), 0)
            if not x.shape[0]:
                continue
            x[:, 15:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 15:] > self.confidence_threshold).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
            else:
                conf, j = x[:, 15:].max(1, keepdim=True)
                x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > self.confidence_threshold]
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 15:16] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, self.iou_threshold) 
            if merge and (1 < n < 3E3):
                iou = box_iou(boxes[i], boxes) > self.iou_threshold
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break 
        return output

    def __del__(self):
        self.destroy()

    