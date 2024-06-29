from fileinput import filename
import xml.etree.cElementTree as ET
import argparse, os, glob
from numpy import random

def add_annotations(xml, train_dest_file, val_dest_file):
    def parse_point(point_string):
        xy = point_string.split(",")
        return [round(float(c)) for c in xy]

    root = ET.fromstring(xml)
    annotations = root.findall(f"./image")
    for annotation in annotations:
        attributes = annotation.attrib        
        if "subset" not in attributes or attributes["subset"] == "Train":
            dest_file = train_dest_file
        elif attributes["subset"] == "Validation":
            dest_file = val_dest_file
        else:
            continue
        
        filename = attributes["name"]
        print("#", filename, file=dest_file)

        for child in annotation:
            ignore = False
            if child.attrib["label"] == "dpg_partial":
                ignore = True
                
            top, left = int(attributes["height"]), int(attributes["width"])
            bottom, right = 0,0
            for point in child:
                x,y = parse_point(point.attrib["points"])
                if(x < left):
                    left = x
                if(y < top):
                    top = y
                if(x > right):
                    right = x
                if(y > bottom):
                    bottom = y

            print(left + 1, top + 1, right - left, bottom - top, file=dest_file, end=" ") # voc format starts indexes with 1
            print(*["-1.0" for _ in range(15)], file=dest_file, end=" ")
            print("0.0" if ignore else "1.0", file=dest_file)

def add_background_imgs(train_dest_file, background_dir, p):
    if not os.path.exists(background_dir):
        return
    img_files = glob.glob(f'{background_dir}/**/*.bmp', recursive=True) + glob.glob(f'{background_dir}/**/*.jpg', recursive=True)
    for f_path in img_files:
        f_path = f_path[len(background_dir):]
        if random.uniform(0,1) < p:
            print("#", f"../Background/{f_path}", file=train_dest_file)
    

def main():
    parser = argparse.ArgumentParser(description="Converts Cvat annotation file to two dlib annotation files")
    parser.add_argument("base_path", help="path or glob to data")
    parser.add_argument("-p", "--extra_background_keep_ratio", default=0.1, help="relative freq of extra images in extra folder to keep")
    args = parser.parse_args()
    
    cvat_file_name = f"{args.base_path}/annotations.xml"
    train_dest_name = f"{args.base_path}/train/label.txt"
    val_dest_name = f"{args.base_path}/val/label.txt"
    
    background_dir = f"{args.base_path}/WIDER_train/Background/"

    os.makedirs(os.path.dirname(train_dest_name), exist_ok=True)
    os.makedirs(os.path.dirname(val_dest_name), exist_ok=True)

    with open(cvat_file_name) as f:
        annotation_xml = f.read()
    with open(train_dest_name, "w") as train_dest_file, open(val_dest_name, "w") as val_dest_file:
        add_annotations(annotation_xml, train_dest_file, val_dest_file)
        add_background_imgs(train_dest_file, background_dir, args.extra_background_keep_ratio)
main()