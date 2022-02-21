from yolov5_ts_detector import Yolov5TSDetector
import yaml
import os
import cv2

def ReadYaml(file_path):
    with open(file_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.CLoader)
    return config

config = ReadYaml("config.yaml")
yolov5_ts = Yolov5TSDetector(config)

img  = cv2.imread("person.jpg")
yolov5_ts.detect(img)
