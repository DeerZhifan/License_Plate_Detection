# -*- coding: utf-8 -*-

from cv2 import dnn
import cv2
import os
import time


class Detection(object):
    """车牌框定位"""
    def __init__(self, width, height, image_path, prototxt_path, caffemodel_path, scale_facotr=0.007843, mean=127.5,):
        """初始化参数"""
        self.width = width
        self.height = height
        self.image_path = image_path
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.scale_factor = scale_facotr
        self.mean = mean

        """读取图片"""
        self.image = cv2.imread(self.image_path)

    def crop_image(self):
        """裁剪图片"""
        cols = self.image.shape[1]
        rows = self.image.shape[0]
        wh_ratio = self.width / float(self.height)

        if cols / float(rows) > wh_ratio:
            crop_size = (int(rows * wh_ratio), rows)
        else:
            crop_size = (cols, int(cols / wh_ratio))

        xLeftBottom = int((cols - crop_size[0]) / 2)
        yLeftBottom = int((rows - crop_size[1]) / 2)
        xRightTop = int(xLeftBottom + crop_size[0])
        yRightTop = int(yLeftBottom + crop_size[1])
        self.image = self.image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]

        cols = self.image.shape[1]
        rows = self.image.shape[0]

        return cols, rows

    def get_detections(self):
        """车牌框检测"""
        blob = dnn.blobFromImage(self.image, self.scale_factor, (self.width, self.height), self.mean, swapRB=True)
        net = dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
        net.setInput(blob)
        start_time = time.time()
        detections = net.forward()
        # print("Cost time: {:.3f}s".format(time.time() - start_time))
        return detections

    def bounding_box_fileter_by_wh_ratio(self, detections, cols, rows):
        """筛选出长宽比不大于0.6的目标框"""
        object_info = {}
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])
                xLeftBottom = int(detections[0, 0, i, 3] * cols) - 6
                yLeftBottom = int(detections[0, 0, i, 4] * rows) - 6
                xRightTop = int(detections[0, 0, i, 5] * cols) + 4
                yRightTop = int(detections[0, 0, i, 6] * rows) + 8

                height, width = yRightTop - yLeftBottom, xRightTop - xLeftBottom
                print(height / width)
                if height / width <= 0.6:
                    object_info[i] = [confidence, class_id, yRightTop, yLeftBottom, xRightTop, xLeftBottom]
        print(object_info)
        return object_info

    def bounding_box_filter_by_width(self, object_info):
        """通过宽度筛选出较小的目标框"""
        key = 0
        if len(object_info) >= 1:
            temp = float("inf")
            for i in object_info.keys():
                bounding_box_width = object_info[i][4] - object_info[i][5]
                if bounding_box_width < temp:
                    temp = bounding_box_width
                    key = i
        # print(key, object_info.keys())
        if key in object_info.keys():
            return object_info[key]
        return None

    def get_bounding_box(self):
        """获取目标框坐标位置及对应置信度"""
        detections = self.get_detections()
        cols, rows = self.crop_image()
        object_info = self.bounding_box_fileter_by_wh_ratio(detections, cols, rows)
        bounding_box = self.bounding_box_filter_by_width(object_info)

        return bounding_box

    def add_bounding_box(self):
        """在图中添加目标框及置信度"""
        class_names = ["Background", "License"]
        bounding_box = self.get_bounding_box()
        if bounding_box is not None:
            confidence, class_id, yRightTop, yLeftBottom, xRightTop, xLeftBottom = tuple(bounding_box)
            # 添加目标框
            cv2.rectangle(self.image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0,255,0), 2)
            # 添加置信度
            label = class_names[class_id] + " : " + str(confidence)
            cv2.putText(self.image, label, (xLeftBottom-5, yLeftBottom-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        return self.image


if __name__ == "__main__":
    width, height = 1024, 720
    prototxt_path, caffemodel_path = "../resources/MobileNetSSD_test.prototxt", "../resources/lpr.caffemodel"
    test_dir = "../27490"
    for f in os.listdir(test_dir):
        if f.endswith(".jpg"):
            image_path = test_dir + "/" + f
            detector = Detection(width, height, image_path, prototxt_path, caffemodel_path)
            image = detector.add_bounding_box()
            cv2.imshow("test", image)
            cv2.waitKey(0)


