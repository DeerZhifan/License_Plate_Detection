# -*- coding: utf-8 -*-

from algo.license_detection import Detection
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class AddWaterMark(object):
    """为图片添加水印"""
    def __init__(self, image, image_name, watermark, bounding_box):
        """初始化参数"""
        self.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.image_name = image_name
        self.watermark = watermark
        self.yRightTop = bounding_box[0]
        self.yLeftBottom = bounding_box[1]
        self.xRightTop = bounding_box[2]
        self.xLeftBottom = bounding_box[3]

    def resize_watermark(self):
        """缩放水印"""
        width = self.xRightTop - self.xLeftBottom + 15
        height = self.yRightTop - self.yLeftBottom + 15
        watermark = self.watermark.resize((width, height), Image.ANTIALIAS)
        return watermark

    def add_watermark(self):
        """添加水印"""
        watermark = self.resize_watermark()
        self.image.paste(watermark, (self.xLeftBottom - 10, self.yLeftBottom - 5))
        plt.imshow(self.image)
        plt.show()

    def save_image(self):
        """保存照片"""
        self.image.save("C:\\Users\\ABC\\PycharmProjects\\License_Plate_Detection\\result\\{:}".format(self.image_name), "JPEG")


if __name__ == "__main__":
    image_path = "test.jpg"
    image_name = "test.jpg"
    watermark_path = "../resources/watermark.png"
    width, height = 1024, 720
    prototxt_path, caffemodel_path = "../resources/MobileNetSSD_test.prototxt", "../resources/lpr.caffemodel"
    detector = Detection(width, height, image_path, prototxt_path, caffemodel_path)
    bounding_box = detector.get_bounding_box()
    if bounding_box is not None:
        bounding_box = bounding_box[2:6]
        image = detector.crop_image()
        watermark = Image.open(watermark_path)
        engine = AddWaterMark(image, image_name, watermark, bounding_box)
        engine.add_watermark()
        engine.save_image()





