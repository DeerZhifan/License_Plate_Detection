# -*- coding: utf-8 -*-
from algo.license_detection import Detection
from algo.add_watermark import AddWaterMark
from PIL import Image
import os
import time


class Main(object):
    """车牌替换主程序入口"""
    def __init__(self, image_path, image_name, watermark_path, width, height, prototxt_path, caffemodel_path):
        """初始化参数"""
        self.image_path = image_path
        self.image_name = image_name
        self.watermark_path = watermark_path
        self.width = width
        self.height = height
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path

    def main(self):
        """执行器"""
        detector = Detection(self.width, self.height, self.image_path, self.prototxt_path, self.caffemodel_path)
        bounding_box = detector.get_bounding_box()
        if bounding_box is not None:
            bounding_box = bounding_box[2:6]
            image = detector.crop_image()
            watermark = Image.open(self.watermark_path)
            engine = AddWaterMark(image, self.image_name, watermark, bounding_box)
            engine.add_watermark()
            engine.save_image()
        else:
            image = Image.open(self.image_path)
            image.save("C:\\Users\\ABC\\PycharmProjects\\License_Plate_Detection\\result\\{:}".format(self.image_name), "JPEG")


if __name__ == "__main__":
    test_dir = "./benz"
    watermark_path = "./resources/watermark.png"
    width, height = 1024, 720
    prototxt_path, caffemodel_path = "./resources/MobileNetSSD_test.prototxt", "./resources/lpr.caffemodel"
    time_recorde  =[]
    for image_name in os.listdir(test_dir):
        start_time = time.time()
        if image_name.endswith(".jpg"):
            image_path = test_dir + "/" + image_name
            engine = Main(image_path, image_name, watermark_path, width, height, prototxt_path, caffemodel_path)
            engine.main()
            end_time = time.time()
            cost_time = end_time - start_time
            time_recorde.append(cost_time)
            print(image_name, "Cost time: {:.3f}s".format(cost_time))
    print("average time : {}".format(sum(time_recorde) / len(time_recorde)))