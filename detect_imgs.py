"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys

import cv2

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.config import fd_config

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="slim", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.6, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument("-p", '--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
input_width = args.input_size

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

result_path = f"./detect_imgs_results-{args.net_type}-{input_width}"
label_path = "./models/voc-model-labels.txt"
test_device = args.test_device

candidate_size = len(fd_config.priors)

class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':
    model_path = f"models/pretrained/dpg-slim-{input_width}.pth"
    # model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif args.net_type == 'RFB':
    model_path = f"models/pretrained/dpg-RFB-{input_width}.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)
    
if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(args.path)
sum_found = 0
for file_path in listdir:
    img_path = os.path.join(args.path, file_path)
    orig_image = cv2.imread(img_path)
    if orig_image is None or not orig_image.data:
        print(f"Failed to read {img_path}")
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, args.threshold)
    n_found = boxes.size(0)
    sum_found += n_found
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(orig_image, (x1, y1, x2-x1, y2-y1), (0, 0, 255), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{probs[i]:.2f}"
        # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(result_path, file_path), orig_image)
    if n_found > 0:
        print(f"Found {n_found} faces in {img_path}. The output image is {result_path}")
print(f"Total found: {sum_found}")
