from genericpath import exists
import logging
import os
import pathlib
from random import randint
from numpy import random
import xml.etree.ElementTree as ET
from ..transforms.transforms import intersect

import cv2
import numpy as np
from math import log10

os.makedirs("voc_augments/tmp/", exist_ok= True)
debug_image_dir = "voc_augments/tmp/"
debug_image_count = 0

class VOCDataset:

    def __init__(self, root, predict_transform, target_transform, train_augmentation=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.predict_transform = predict_transform
        self.target_transform = target_transform
        self.train_augmentation = train_augmentation
        
        self.is_test = is_test
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
            self.bgs = []
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
            bg_sets_file = self.root / "ImageSets/Main/background.txt"
            self.bgs = VOCDataset._read_image_ids(bg_sets_file, optional=True)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
                                'face')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        global debug_image_count

        if not self.is_test and self.bgs and random.randint(10) == 0:
            n_faces = int(log10(random.randint(10000))) + 1
            image, boxes, labels = self._get_item_synthetic(n_faces)
        else:
            image, boxes, labels = self._get_item_normal(index)
        if self.train_augmentation and not self.is_test:
            image, boxes, labels = self.train_augmentation(image, boxes, labels)
            if True:
                file_count = debug_image_count
                #print(f"writing debug image {debug_image_dir}/{file_count}.jpg: {boxes}")
                cv2.imwrite(f"{debug_image_dir}/{file_count}.jpg", image)
                pass
            debug_image_count += 1
        if self.predict_transform:
            image, boxes, labels = self.predict_transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
           
        return image, boxes, labels
    
    def _get_item_normal(self, index):
        image_id = self.ids[index]

        boxes, labels, _ = self._get_annotation(image_id)
        image = self._read_image(image_id)
        return image, boxes, labels
        
    def _get_item_synthetic(self, n_copies):
        image = self._read_background_image()
        labels = []
        boxes = np.empty((0, 4))
        return self._perform_copy_paste(image, boxes, labels, n_copies)

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.train_augmentation and not self.is_test:
            image, _, _ = self.train_augmentation(image)
        if self.predict_transform:
            image, _, _ = self.predict_transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file, optional=False):
        ids = []
        try:
            with open(image_sets_file) as f:
                for line in f:
                    ids.append(line.rstrip())
        except IOError as ex:
            if not optional:
                raise
            else:          
                print(f"Could not open {image_sets_file} continue without background imgs")
                print(ex)
        return ids
    
    def _perform_copy_paste(self, image, boxes, labels, n_copies):
        labels = list(labels)
        boxes = list(boxes)
        for _ in range(n_copies):
            for _ in range(50):
                copy_face, label = self._read_random_face()
                h, w, _ = image.shape
                box_h, box_w, _ = copy_face.shape
                aspect_ratio = box_h / box_w
                box_w = random.randint(int(0.07 * w), int(0.4 * w)),
                box_h = int(box_w * aspect_ratio)
                copy_face = cv2.resize(copy_face, dims=(box_w, box_h))

                random_x, random_y = random.randint(w), random.randint(h)
                x1, y1, x2, y2 = random_x, random_y, random_x + box_w, random_y + box_h
                new_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                if x2 < w and y2 < h:
                    if len(boxes) > 0 and intersect(np.array(boxes, dtype=np.float32).reshape(-1, 4), new_box).max() > 0:
                        continue
                    image[y1:y2, x1:x2, :] = copy_face
                    boxes.append(new_box)
                    labels.append(label)
                    break
        boxes = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else None
        labels =  np.array(labels, dtype=np.int64) if len(labels) > 0 else None
        return image, boxes, labels

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))
    
    def _read_random_face(self):
        assert(not self.is_test)
        
        index = random.randint(len(self.ids))
        copy_img, new_boxes, new_labels = self._get_item_normal(index)
        copy_img, new_boxes, new_labels = self.train_augmentation(copy_img, new_boxes, new_labels)
            
        chosen = random.randint(len(new_labels))
        box, label = np.array(new_boxes[chosen, :], dtype=int), new_labels[chosen]
        copy_face = copy_img[box[1]:box[3], box[0]:box[2], :]
        return copy_face, label

    def _read_background_image(self):
        assert(not self.is_test)
                
        image_id = random.choice(self.bgs)
        image_file = self.root / f"Background/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train_augmentation:
            image, _, _ = self.train_augmentation(image, None, None)
        return image

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
