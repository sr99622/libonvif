import numpy as np
#import cv2
from PIL import Image, ImageDraw

class Echo:
    def __init__(self, arg):
        print("echo.__init__")

        # parse the initialization string
        unpacked_args = arg[0].split(",")
        for line in unpacked_args:
            key_value = line.split("=")
            print("key  ", key_value[0])
            print("value", key_value[1])

    def __call__(self, arg):
        print("echo.__call__")
        img = arg[0][0]
        print('img shape', img.shape)
        pts = arg[1][0]
        print("pts", pts)
        rts = arg[2][0]
        print("rts", rts)

        image = Image.fromarray(img.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.line([(0, 0), (1000, 1000)], fill=(255, 255, 255), width=10)

        img = np.asarray(image)

        # Possible return arguments

        #return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)       # return a modified image
        return img
        #return pts       # return a modified pts
        #return False     # record trigger argument

        #return (img, pts, False)
        #return (img, pts)
        #return (img, False)
