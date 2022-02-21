import torch
import numpy as np
import typing
from det_object import DetObject, DetBbox
from yolov5_utils import non_max_suppression, scale_coords, xyxy2xywh, letterbox


class Yolov5TSDetector:
    def __init__(self, config):
        self.device = config["device"]
        self.model_path = config["model_path"][self.device]
        self.img_width = config["img"]["width"]
        self.img_height = config["img"]["height"]
        self.class_names = config["class_names"]
        self.score_threshold = config["nms"]["score_threshold"]
        self.iou_threshold = config["nms"]["iou_threshold"]
        self.init_model()

    def init_model(self):
        self.model = torch.jit.load(self.model_path)

        tensor = self.preprocess_img(None)[0]
        with torch.no_grad():
            self.model(tensor)
            self.model(tensor)

    def preprocess_img(self, img):
        if img is None:
            tensor = torch.zeros(1, 3, self.img_height, self.img_width)
            ratio = (1, 1)
            pad = (0.0, 0.0)
        else:
            img, ratio, pad = letterbox(
                img,
                new_shape=(self.img_height, self.img_width),
                auto=False,
            )
            tensor = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            tensor = np.ascontiguousarray(tensor)

            tensor = torch.from_numpy(tensor).to(self.device, non_blocking=True)
            tensor = tensor.float()
            tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor, ratio, pad

    def detect(self, img) -> typing.List[DetObject]:
        tensor, ratio, pad = self.preprocess_img(img)
        with torch.no_grad():
            pred = self.model(tensor)[0]
        pred = non_max_suppression(
            pred.cpu(), self.score_threshold, self.iou_threshold
        )
        output = []
        for det in pred:
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(
                    tensor.shape[2:], det[:, :4], img.shape, ratio_pad=(ratio, pad)
                ).round()

                for *xyxy, score, cls_id in reversed(det):
                    xyxy = torch.tensor(xyxy).view(1, 4)
                    xywh = xyxy2xywh(xyxy)
                    xc, yc, w, h = map(float, (xywh / gn).view(-1).tolist())
                    bbox = DetBbox(xc, yc, w, h)
                    print(self.class_names[int(cls_id)])
                    obj = DetObject(bbox, score, self.class_names[int(cls_id)])
                    output.append(obj)

        return output
