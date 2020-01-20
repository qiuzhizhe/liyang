from model.detect_model import yolov3_detect
import cv2
import time
import os
single_img_path =os.getcwd()+'/imgs/***.jpg'#单张图片测试根目录
input_images_path  = os.getcwd()+'/imgs'#批量测试图片目录根目录

yolo_model = yolov3_detect()
def file_name(file_dir):
    L=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames :
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(dirpath, file))
    return L

#批量测试
def test_imgs():
    file_list = file_name(input_images_path)
    for j in range(len(file_list)):
        path = file_list[j]
        img_ori = cv2.imread(path)
        num = yolo_model.detect_and_save(img_ori, path)

#单张图片测试
def test_single_img():
    img_ori = cv2.imread(single_img_path)
    num = yolo_model.detect_bbox(img_ori, single_img_path)

