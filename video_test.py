# coding: utf-8

from __future__ import division, print_function
import cv2
from model.detect_model import yolov3_detect
import os
yolo_model = yolov3_detect()

input_video_path=os.getcwd()+'/test_video/10.mp4'#测试视频路径
save_video_path= os.getcwd()+'/video_result/video_result.mp4'#视频结果保存路径
save_video = True#是否保存视频结果
new_size =[608,608]
vid = cv2.VideoCapture(input_video_path)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))
fram=10

if save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(save_video_path, fourcc, 1, (new_size[0], new_size[1]))

for i in range(video_frame_cnt):
    if i%fram != 0:
        continue
    ret, img_ori = vid.read()
    img_ori=yolo_model.video_test_detect(img_ori)
    img_ori = cv2.resize(img_ori, tuple(new_size))
    cv2.namedWindow("image", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow("image",608,608)
    cv2.imshow('image', img_ori)
    if save_video:
        videoWriter.write(img_ori)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
if save_video:
    videoWriter.release()
