from trt_model import TRTModel
from .yolov5_utils import *
import torchvision
import copy
import cv2
import torch
import time



class YOLOV5(TRTModel):
    def __init__(self, config):
        super().__init__(config)
        # self.onnx = config["face_detection"]["onnx"]
        # self.engine = config["face_detection"]["engine"]
        # self.fp16_mode = config["face_detection"]["fp16_mode"]
        self.output_shape = [1,25200,16]
        self.input_shape = [640,640]
        self.stride_max = 32
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
    
    def inference(self, img):
        # img = img.numpy()
        pred = self(img)
        return pred

    def pre_process(self, orgimg):
        img = letterbox(orgimg, [640, 640], stride=32, auto=False)[0]
        # img = cv2.resize(orgimg, (640,640))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to("cuda")
        fp16 = False
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im
        # img0 = copy.deepcopy(orgimg)
        # h0, w0 = orgimg.shape[:2]
        # r = self.input_shape[0]/ max(h0, w0) 
        # if r != 1:
        #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #     img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # imgsz = check_img_size(self.input_shape[0], s=self.stride_max) 
        # img = letterbox(img0, new_shape=imgsz,auto=False)[0]
        # img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        # img = torch.from_numpy(img)
        # img = img.float()
        # img /= 255.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        # return img,orgimg

    def nms(self, prediction):
        nc = prediction.shape[2] - 15  # number of classes
        xc = prediction[..., 4] > self.confidence_threshold  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            # if labels and len(labels[xi]):
            #     l = labels[xi]
            #     v = torch.zeros((len(l), nc + 15), device=x.device)
            #     v[:, :4] = l[:, 1:5]  # box
            #     v[:, 4] = 1.0  # conf
            #     v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            #     x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, landmarks, cls)
            if multi_label:
                i, j = (x[:, 15:] > self.confidence_threshold).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 15:].max(1, keepdim=True)
                x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > self.confidence_threshold]

            # Filter by class
            # if classes is not None:
            #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            agnostic=False
            c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iou_threshold)  # NMS
            #if i.shape[0] > max_det:  # limit detections
            #    i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > self.iou_threshold  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    def post_process(self, img,orgimg,pred):
        pred = self.nms(pred)
        print(pred[0].shape)
        no_vis_nums=0
        results = []
        for i, det in enumerate(pred):
            gn = torch.tensor(orgimg.shape, device='cuda:0')[[1, 0, 1, 0]]
            gn_lks = torch.tensor(orgimg.shape, device='cuda:0')[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
                for j in range(det.size()[0]):
                    xyxy2xywh(det[j, :4].view(1, 4))/gn
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    result = get_bbox(orgimg, xywh, conf, landmarks)
                    results.append(result)
        return results