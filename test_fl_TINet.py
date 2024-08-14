import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.FL_TINet import FL_TINet_body
from utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)


# --------------------------------------------#
#   # When using your own trained model for prediction, you need to modify 3 parameters:
# # The model_path, classes_path, and phi all need to be updated!
# # If you encounter a shape mismatch error, please pay close attention to the following:
# # Ensure that the model_path, classes_path, and phi parameters used during prediction match those used during training
# --------------------------------------------#
class FLTINetApp(object):
    _defaults = {
        "model_path": 'model_data/FL_TINet_weights_voc_ECA.pth',
        # "model_path"        : 'model_data/FL_TINet_weights_voc.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        # -------------------------------#
        #   The type of attention mechanism used
        #   phi = 0 for no attention mechanism
        #   phi = 1 for SE
        #   phi = 2 for CBAM
        #   phi = 3 for MAM
        # -------------------------------#
        "phi": 3,
        "model_image_size": (608, 608, 3),
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": True,

        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Inintialize FL-TINet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   Obtain all classes
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   Obtain all the anchor boxes
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])

    # ---------------------------------------------------#
    #      build model
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   generate FL_TINet
        # ---------------------------------------------------#
        self.net = FL_TINet_body(len(self.anchors[0]), len(self.class_names), self.phi).eval()

        # ---------------------------------------------------#
        #   load wiethts of FL-TINet
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # ---------------------------------------------------#
        #   stablish feature layer decoding tools
        # ---------------------------------------------------#
        self.yolo_decodes = []
        self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        for i in range(2):
            self.yolo_decodes.append(
                DecodeBox(np.reshape(self.anchors, [-1, 2])[self.anchors_mask[i]], len(self.class_names),
                          (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # Set different colors for drawing bounding boxes
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   detect images
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   Here, convert the image to an RGB image to prevent errors during prediction with grayscale images
        # ---------------------------------------------------------#
        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   # Add gray bars to the image to achieve distortion-free resizing.
        #  Alternatively, you can directly resize the image for recognition.
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   Add the batch_size dimension for processing.
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------------#
            #   Input the image into the network for prediction
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            # ---------------------------------------------------------#
            #   Stack the predicted bounding boxes and then apply non-maximum suppression
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            # ---------------------------------------------------------#
            #   If no object is detected, return the original image.
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # ---------------------------------------------------------#
            #   Perform score filtering on the predicted bounding boxes.
            # ---------------------------------------------------------#
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   Before passing the image into the network for prediction, letterbox_image is applied to add gray bars around the image.
            #   Therefore, the generated top_bboxes are relative to the image with gray bars.
            # -----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence,
                                                   nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                top_label = np.array(batch_detections[top_index, -1], np.int32)
                top_bboxes = np.array(batch_detections[top_index, :4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                    top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                               np.array([self.model_image_size[0], self.model_image_size[1]]),
                                               image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
            except:
                pass

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(2):
                    output_list.append(self.yolo_decodes[i](outputs[i]))

                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence,
                                                       nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index, -1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index, :4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3],
                                                                                                    -1)

                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                                   image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
                except:
                    pass
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
