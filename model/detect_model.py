# coding: utf-8
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import cv2
import os
from model.model_read import yolov3
from model.misc_utils import parse_anchors,read_class_names
from model.plot_utils import get_color_table, plot_one_box
model_path=os.getcwd()+'/checkpoint/yolo_3.ckpt'
anchor_path=os.getcwd()+'/data/lift_anchors.txt'
class_name_path = os.getcwd()+'/data/lift.names'
save_result_path=os.getcwd()+'/img_detect_result/'
sourcr_imgpath = os.getcwd()+'/img_path/'

class yolov3_detect(object):

    def __init__(self,restore_path=model_path,num_class=1,new_size=[608,608]):
        self.sess=tf.Session()
        self.num_class=num_class
        self.restore_path =restore_path
        self.anchors = parse_anchors(anchor_path)
        self.classes =  read_class_names(class_name_path)
        self.color_table = get_color_table(num_class)
        self.new_size = new_size
        self.input_data = tf.placeholder(tf.float32, [1, self.new_size[1],self.new_size[0], 3], name='input_data')
        with tf.device('/cpu:0'):
            yolo_model = yolov3(self.num_class, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            self.pred_boxes, self.pred_confs, self.pred_probs = yolo_model.predict(pred_feature_maps)
            saver = tf.train.Saver()
            saver.restore(self.sess, self.restore_path)

    def check(self,box, high):
        return True
        w = (box[0] + box[2]) / 2
        h = (box[1] + box[3]) / 2
        x_ = high / 2
        y_ = high / 2
        wh = (box[2] - x_) * (box[2] - x_) + (box[3] - y_) * (box[3] - y_)

        if wh > x_ * x_ * 0.8:
            return False
        return True

    def letterbox_resize(self,img, new_width, new_height, interp=0):
        '''
        Letterbox resize. keep the original aspect ratio in the resized image.
        '''
        ori_height, ori_width = img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded, resize_ratio, dw, dh
    def py_nms(self,boxes, scores, max_boxes=50, iou_thresh=0.5):
        """
        Pure Python NMS baseline.

        Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                          exact number of boxes
                   scores: shape of [-1,]
                   max_boxes: representing the maximum of boxes to be selected by non_max_suppression
                   iou_thresh: representing iou_threshold for deciding to keep boxes
        """
        assert boxes.shape[1] == 4 and len(scores.shape) == 1

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep[:max_boxes]
    def cpu_nms(self,boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
        """
        Perform NMS on CPU.
        Arguments:
            boxes: shape [1, 10647, 4]
            scores: shape [1, 10647, num_classes]
        """

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, num_classes)
        # Picked bounding boxes
        picked_boxes, picked_score, picked_label = [], [], []

        for i in range(num_classes):
            indices = np.where(scores[:, i] >= score_thresh)
            filter_boxes = boxes[indices]
            filter_scores = scores[:, i][indices]
            if len(filter_boxes) == 0:
                continue
            # do non_max_suppression on the cpu
            indices = self.py_nms(filter_boxes, filter_scores,
                             max_boxes=max_boxes, iou_thresh=iou_thresh)
            picked_boxes.append(filter_boxes[indices])
            picked_score.append(filter_scores[indices])
            picked_label.append(np.ones(len(indices), dtype='int32') * i)
        if len(picked_boxes) == 0:
            return None, None, None

        boxes = np.concatenate(picked_boxes, axis=0)
        score = np.concatenate(picked_score, axis=0)
        label = np.concatenate(picked_label, axis=0)

        return boxes, score, label

    '''输入图片，输出人员数量'''
    def detect(self,input,letterbox_resize_able = True,new_size=[608,608]):
        #img_ori = img_ori[bottom:top, left:right]
        with tf.device('/cpu:0'):
            if letterbox_resize_able:
                img, resize_ratio, dw, dh = self.letterbox_resize(input, new_size[0], new_size[1])
            else:
                height_ori, width_ori = input.shape[:2]
                img = cv2.resize(input, tuple(new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h = img.shape[0];
            w_start = (img.shape[1] - img.shape[0]) // 2
            w_end = img.shape[1] - w_start
            img = img[0:img.shape[0], w_start:w_end]
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            pred_boxes_, pred_confs_, pred_probs_ = self.sess.run([self.pred_boxes, self.pred_confs, self.pred_probs],
                                                                  feed_dict={self.input_data: img})
            pred_scores = pred_confs_ * pred_probs_
            boxes_, scores_, labels_ = self.cpu_nms(pred_boxes_, pred_scores, self.num_class, max_boxes=200,
                                                    score_thresh=0.5, iou_thresh=0.45)
            num = len(boxes_)
            return num

    def detect_plot_bboxf(self,input,letterbox_resize_able = True,new_size=[608,608]):

        with tf.device('/cpu:0'):
            h = input.shape[0];
            w_start = (input.shape[1] - input.shape[0]) // 2
            w_end = input.shape[1] - w_start
            input = input[0:input.shape[0], w_start:w_end]

            if letterbox_resize_able:
                img, resize_ratio, dw, dh = self.letterbox_resize(input, new_size[0], new_size[1])
            else:
                height_ori, width_ori = input.shape[:2]
                img = cv2.resize(input, tuple(new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            pred_boxes_, pred_confs_, pred_probs_ = self.sess.run([self.pred_boxes, self.pred_confs, self.pred_probs],
                                                                  feed_dict={self.input_data: img})
            pred_scores = pred_confs_ * pred_probs_
            boxes_, scores_, labels_ = self.cpu_nms(pred_boxes_, pred_scores, self.num_class, max_boxes=200,
                                                    score_thresh=0.5, iou_thresh=0.45)

            # rescale the coordinates to the original image
            if letterbox_resize_able:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(new_size[1]))
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                if self.check(boxes_[i], h):
                    plot_one_box(input, [x0, y0, x1, y1],
                                 label=self.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                                 color=self.color_table[labels_[i]])
        return len(boxes_),input

    '''输入图片和图片路径，输出人员数量、图片检测结果自动保存到img_detect_result文件夹下'''
    def detect_and_save(self,input,path,letterbox_resize_able = True,new_size=[608,608]):
        num,img=self.detect_plot_bboxf(input,letterbox_resize_able,new_size)
        cv2.imwrite(save_result_path +path.split('\\')[1], img)
        return num

    def video_test_detect(self,input,letterbox_resize_able = True,new_size=[608,608]):

        num, img = self.detect_plot_bboxf(input, letterbox_resize_able, new_size)
        return img







