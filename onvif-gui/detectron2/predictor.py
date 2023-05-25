import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

class Predictor:

    def __init__(self, cfg, fp16=False):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        self.fp16 = fp16
        if self.fp16:
            print("Enabling model for fp16")
            self.model.half()
        self.first_pass = True

    def __call__(self, image):
        with torch.no_grad():
            height, width = image.shape[:2]
            aspect = float(width) / float(height)
            if aspect > 1.0:
                test_width = min(width, self.cfg.INPUT.MAX_SIZE_TEST)
                test_height = int(test_width / aspect)
            else:
                test_height = min(height, self.cfg.INPUT.MAX_SIZE_TEST)
                test_width = int(test_height * aspect)

            if (height, width) != (test_height, test_width):
                image = cv2.resize(image, (test_width, test_height))
                if self.first_pass:
                    print("model test size height:", test_height, " width:", test_width)
            else:
                if self.first_pass:
                    print("direct image input into model without resize")

            self.first_pass = False
            image = torch.as_tensor(image).cuda()
            image = torch.permute(image, (2, 0, 1))

            if self.fp16:
                image = image.type(torch.float16)
            else:
                image = image.type(torch.float32)

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
