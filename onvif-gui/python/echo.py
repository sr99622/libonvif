import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])


class Echo:
    def __init__(self, arg):
        print("echo.__init__")

        # parse the initialization string
        unpacked_args = arg[0].split(",")
        for line in unpacked_args:
            key_value = line.split("=")
            print("key  ", key_value[0])
            print("value", key_value[1])
        
        
        self.min_size = 800
        self.threshold = 0.35
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True,  min_size=self.min_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(self.device)



    def __call__(self, arg):
        print("echo.__call__")
        img = arg[0][0]
        print('img shape', img.shape)
        pts = arg[1][0]
        print("pts", pts)
        rts = arg[2][0]
        print("rts", rts)

        #img = arg[0][0]
        print("test 1")
        tensor = transform(img).to(self.device)
        print("test 2")
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)

        scores = outputs[0]['scores'].detach().cpu().numpy()
        labels = outputs[0]['labels'].detach().cpu().numpy()
        boxes = outputs[0]['boxes'].detach().cpu().numpy()
        labels = labels[np.array(scores) >= self.threshold]
        boxes = boxes[np.array(scores) >= self.threshold].astype(np.int32)
        boxes = boxes[np.array(labels) == 1]


        image = Image.fromarray(img.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.line([(0, 0), (1000, 1000)], fill=(255, 0, 255), width=10)
        img = np.asarray(image)

        # Possible return arguments

        #return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)       # return a modified image
        return img
        #return pts       # return a modified pts
        #return False     # record trigger argument

        #return (img, pts, False)
        #return (img, pts)
        #return (img, False)
