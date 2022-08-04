
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

    def inference(self, img):
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

    def nms(self, prediction):
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
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 15:16] * max_wh
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

    def post_process(self, img,orgimg,pred):
        pred = self.nms(pred)
        no_vis_nums=0
        results = []
        for i, det in enumerate(pred):
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    result = get_bbox(orgimg, xywh, conf, landmarks)
                    results.append(result)
        face_objects = []
        for obj in results:
            bbox = [obj[0], obj[1], obj[2], obj[3]]
            lms = [[obj[4], obj[5]],[obj[6], obj[7]],[obj[8], obj[9]],[obj[10], obj[11]],[obj[12], obj[13]]]
            face_object = {"bbox": bbox, "landmarks": lms, "score": obj[14]}
            face_objects.append(face_object)
        return face_objects

    def __del__(self):
        self.destroy()

    