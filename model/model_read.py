# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf

slim = tf.contrib.slim

from model.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer


class yolov3(object):

    def __init__(self,
                 class_num,
                 anchors,
                 use_label_smooth=False,
                 use_focal_loss=False,
                 batch_norm_decay=0.999,
                 weight_decay=5e-4):
        """
        yolov3 class
        :param class_num: 类别数目
        :param anchors: anchors，一般来说是9个anchors
        :param use_label_smooth: 是否使用label smooth，默认为False
        :param use_focal_loss: 是否使用focal loss，默认为False
        :param batch_norm_decay: BN的衰减系数
        :param weight_decay: 权重衰减系数
        """
        # self.anchors = [[10, 13], [16, 30], [33, 23],
        # [30, 61], [62, 45], [59,  119],
        # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay

    def forward(self, inputs, is_training=False, reuse=False):
        """
        进行正向传播，返回的是若干特征图
        :param inputs: shape: [N, height, width, channel]
        :param is_training:
        :param reuse:
        :return:
        """

        # 获取输入图片的高度height和宽度width
        # the input img_size, form: [height, width]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]

        # batch normalization的相关参数
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # slim的arg scope，可以简化代码的编写，共用一套参数设置
        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                # DarkNet 的主体部分，主要作用是提取图片中的各种特征信息。
                # 这里可以获取三张特征图，分别取自DarkNet的三个不同的阶段。
                # 每一个阶段对应于不同的特征粒度，结合更多的特征可以增强模型的表达能力。
                # 理论上来说特征提取网络也可以采用其他的网络结构，但是效果可能会有所差异。
                # 如果输入图片的尺寸为[416, 416]，则三张特征图的尺寸分别为
                # route_1 : [1, 52, 52, 256]
                # route_2 : [1, 26, 26, 512]
                # route_3 : [1, 13, 13, 1024]
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                # 根据前面的特征图，进行特征融合操作，这样可以提供更多的信息。
                with tf.variable_scope('yolov3_head'):
                    # 使用YOLO_block函数来处理得到的特征图，并返回两张特征图。
                    # 本质上，YOLO_block函数仅仅包含若干层卷积层。
                    # 其中，inter1的作用是用来后续进行特征融合，net的主要作用是用以计算后续的坐标和概率等信息。
                    inter1, net = yolo_block(route_3, 512)

                    # 进行依次卷积，主要是为了进行通道数目调整
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    # 进行一次卷积，调整通道数目为256。并进行上采样，这里的上采样主要是用最近邻插值法。
                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    # 进行特征的融合，这里是通道的融合
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    # 下面的和前面的过程是一致的，不再赘述。
                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')
            # 将三张特征图返回，shape分别如下：（输入图片尺寸默认为[416, 416])
            # feature_map_1: [1, 13, 13, 255]
            # feature_map_2: [1, 26, 25, 255]
            # feature_map_3: [1, 52, 52, 255]
            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        """需要注意的是，我们在下面的代码中会经常涉及到height， width这两个概念，在YOLOv3中，height表示的是竖直方向，
            width表示的是水平方向，同样，x的方向也表示的是水平方向，y的方向是竖直方向"""
        # NOTE: size in [h, w] format! don't get messed up!
        # 获取特征图的尺寸信息，顺序为： [height, width]
        grid_size = tf.shape(feature_map)[1:3]  # [13, 13]

        # the downscale ratio in height and weight
        # 计算此特征图和原图片的缩放尺寸，顺序为： [height, width]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        # 将anchors映射到特征图上,主要是大小上的映射,将anchors的尺寸分别处以下采样倍数即可
        # 需要注意的是，anchors的顺序是[width, height]！所因此下面代码中ratio的下标是反的.
        # 所以计算出的rescaled_anchors的顺序也是[width, height]。
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        # 将特征图reshape一下,主要是将最后一个通道进行分离
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y].
        # 需要注意的是这里的center_x, 和center_y的方向表示,center_x表示的是
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        # 沿着最后一个数据通道进行分离,分别分离成2, 2, 1, class_num的矩阵.
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)

        # 将box的中心数据限制在（0， 1）的范围之内，
        # 因为YOLO将图片分成了一个一个的格子，每一个格子的长宽被设置为1，这里的中心数据本质上是相对于格子左上角的偏移。
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        # grid_x: [0, 1, 2, ..., width - 1]
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        # grid_y: [0, 1, 2, ..., height - 1]
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        # grid_x: [[0, 1, 2, ..., width - 1],
        #          [0, 1, 2, ..., width - 1],
        #          ...
        #          [0, 1, 2, ..., width - 1]]
        # grid_y: [[0, 0, 0, ..., 0],
        #          [1, 1, 1, ..., 1],
        #          ...
        #          [height - 1, height - 1, height - 1, ..., height - 1]]
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x,
                              (-1, 1))  # [0, 1, 2, .., width - 1, 0, 1, 2, ..width - 1, ......, 0, 1, 2, .. width - 1]
        y_offset = tf.reshape(grid_y,
                              (-1, 1))  # [0, 0, 0, .., 0, 1, 1, 1, ...1, ......, height -1, height -1, .., height - 1]

        # x_y_offset: [[0, 0],
        #              [1, 0],
        #              ...
        #              [width - 1, 0],
        #              [0, 1],
        #              [1, 1],
        #              ...
        #              [width - 1, 1],
        #              ......
        #              [0, height - 1],
        #              [1, height - 1],
        #              ...
        #              [width - 1, height - 1]]
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2] 、[height, width, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        # broadcast机制： [N, height, width, 3, 2] = [N, height, width, 3, 2] + [height, width, 1, 2]
        box_centers = box_centers + x_y_offset

        # rescale to the original image scale
        # 将box的中心重新映射到原始尺寸的图片上。
        # 在前面的代码中，最后一个维度的顺序一直是[width, height]的格式，二ratio的顺序是[height, width]，
        # 因此这是需要对ratio取反遍历，结果的顺序依然是[width, height]。
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        # 和前面的过程一样，这里对box的尺寸进行变换，最后一维度的顺序依然是[width, height]
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        # 一样是将box的尺寸重新映射到原始图片上
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]、[N, height, width, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2], [height, width, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]、 [N, height, width, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]、 [N, height, width, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        #
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        # 将特征图和不同尺寸的anchors相结合，缩放程度大的特征图和大尺寸的anchors相结合，
        # 反之，缩放程度小的特征图和小尺寸的anchors相结合
        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]

        # 利用特征图和其对应的anchors计算每一张特征图的预测回归框，置信程度，分类概率等
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            # 取出每一个特征图对应的所有信息，包括预测回归框，置信程度，分类概率等
            x_y_offset, boxes, conf_logits, prob_logits = result

            # 获得特征图的尺寸，[height, width]
            grid_size = tf.shape(x_y_offset)[:2]

            # 将boxes， 前景置信度，分类概率展开
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example),
            # boxes: [N, 13*13*3, 4] , [N, height * width * anchor_num, 4]
            # conf_logits: [N, 13*13*3, 1], [N, height * width * anchor_num, 1]
            # prob_logits: [N, 13*13*3, class_num], [N, height * width * anchor_num, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            # 对每个特征图的偏移量，boxes，前景置信度，分类概率等进行处理（主要是reshape），得到boxes，前景置信度，分类概率。
            boxes, conf_logits, prob_logits = _reshape(result)

            # 对置信度和概率进行sigmoid处理，保证数值位于0~1之间
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            # 将所有的boxes， 前景置信度，分类概率保存起来
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]、[N, box_num, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]、[N, box_num, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]、[N, box_num, class_num]
        probs = tf.concat(probs_list, axis=1)

        # 接下来处理boxes，我们需要将存储格式为中心加尺寸的box数据变换成左上角和右下角的坐标。
        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        # 返回boxes，前景置信度，以及分类概率
        return boxes, confs, probs

    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''

        # size in [h, w] format! don't get messed up!
        # 获取特征图的尺寸，这里的顺序是[height, width]
        grid_size = tf.shape(feature_map_i)[1:3]

        # the downscale ratio in height and weight
        # 计算下采样的倍数，使用的是原始图片的尺寸除以特征图的尺寸，所以顺序依然是[height, width]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        # N: batch_size
        # 样本数目，或者说batch size，这里转换成了浮点数
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        # 根据特征图和每一个特征图对应的anchors计算预测的Bboxes，每一个框的概率以及每一个框属于前景的概率。
        # 这里返回的第一个参数是每一张特征图上的偏移量。
        # x_y_offset: [height, width, 1, 2]
        # pred_boxes: [N, height, width, 3, 4]
        # pred_conf_logits: [N, height, width, 3, 1]
        # pred_prob_logits: [N, height, width, 3, 80(num_class)]
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        # y true的最后一维的格式是[4, 1, 80, 1],分别表示4位坐标， 1位前景标志位，80个分类标记，1位mix up标记位
        # y_true的最后一个维度的4号位（由0开始计数）上存储的是当前位置是否是一个有效的前景.
        # 如果某一个目标的中心落入框中，则是一个有效的前景，当前位是1，否则当前位置是0.
        # 以13 * 13的特征图为例，object mask的shape是[N, 13, 13, 3, 1] ([N, height, width, 3, 1]).
        object_mask = y_true[..., 4:5]

        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box
        # 根据上面计算出来的有效前景框，提取有效的ground truth前景框的坐标，
        # valid true boxes的shape：[V, 4]， 这里的V表示的是有效的ground truth前景框的数目。
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # shape: [V, 2]
        # 将gt目标框的中心和高度宽度分离成两个矩阵，每个矩阵的shape都是[V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]

        # shape: [N, 13, 13, 3, 2]
        # 同样，我们将特征图预测的每个位置的目标框的中心坐标和高度宽度提取出来。
        # pred boxes的最后一个维度是[2, 2, 1, 80, 1],
        # 分别表示预测的边界框的中心位置（2），预测的边界框的高度宽度（2），预测的边界框的前景置信度（1），分类置信度（80），mixup权重（1）
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        # 计算在每个位置上，每个预测的目标框和V个gt目标框之间的iou，返回相对应的矩阵。
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        # 这一步相当于是为每一个预测的目标框匹配一个最佳的iou。
        # 当然有些预测的目标框是不和任何的gt目标框相交的，此时它的最佳匹配的iou就是0.
        best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask
        # 计算出那些和任何一个gt目标边界框的iou都小于0.5的预测目标框的标记。
        # 虽然某些框和目标有一定的重叠，但是重叠部分不是很大，我们忽略掉这些框
        # shape：[N, 13, 13, 3]
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        # 扩展出最后一个维度，这个ignore mask后面计算损失会用到
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        # 计算gt目标框和预测的目标框相对于网格坐标的偏移量。
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        # 计算gt目标框和预测的目标框相对于anchors的大小缩放量
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors

        # for numerical stability
        # 为了保证数据的稳定性，因为log(0)会趋向于负无穷大，因此将0设置为1，log之后就会变成0，可以看作不影响。
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        # 取对数，这里使用了范围的限制，小于1e-9的会强制变成1e-9，大于1e9的数据会变成1e9。
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        # 对于目标框尺寸的惩罚，尺寸较小的框具有较大的权重。
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############

        # mix_up weight
        # [N, 13, 13, 3, 1]
        # mix up 权重
        mix_w = y_true[..., -1:]

        # shape: [N, 13, 13, 3, 1]
        # 这里计算目标框的中心偏移的损失和高度宽度的损失，这里使用了均方和的方式计算。
        # 从式子中可以看出，我们关注的只有object mask为1的目标，即有效的目标框，其他的目标框就被忽略了。
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # 前景的正样本mask，这里直接使用了object mask，因为这一部分肯定是正确的前景
        conf_pos_mask = object_mask

        # 前景的负样本mask
        # 这里的采样法是没有任何一个gt目标框的中心落入框中，并且和任何一个gt目标框的iou都小于0.5的框作为前景采样的负样本。
        # 这里的iou控制就是使用的ignore mask
        conf_neg_mask = (1 - object_mask) * ignore_mask

        # 使用交叉熵公式计算最后的损失，唯一的区别就是采样的方式，一个是正样本采样，一个是负样本采样
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)

        # TODO: may need to balance the pos-neg by multiplying some weights
        # 二者相加就是最后的前景分类的损失
        conf_loss = conf_loss_pos + conf_loss_neg

        # 是否使用focal loss，默认为False
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            # Focal loss的计算，这不是YOLO的中点，在此省略
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask

        # 将结果和mis up权重相乘，并取均值作为最后的损失标量
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        # 是否使用label smooth，默认为False
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]

        # 分类损失，这里仍然使用的是交叉熵损失。这里还是只对有效的前景框计算损失。最后仍然要和mix up权重相乘
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target,
                                                                           logits=pred_prob_logits) * mix_w
        # 取均值作为最后的分类损失的标量
        class_loss = tf.reduce_sum(class_loss) / N

        # 返回最后的所有损失
        return xy_loss, wh_loss, conf_loss, class_loss

    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''

        # 以下的四个变量分别用来保存四个方面的loss。
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.

        # 对anchors进行分组，因为每一层特征图都对应三个不同尺度的anchors。
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # 对每一张特征图和其对应的真实值以及其对应的anchors计算损失。
        # 一共有三张特征图，故一共存在三个不同尺度的损失。
        # calc loss in 3 scales
        for i in range(len(y_pred)):
            # 分别计算损失
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou


