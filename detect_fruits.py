import json
from pathlib import Path
from typing import Dict

import click
import cv2 as cv
from tqdm import tqdm

def ScaleImage(input_image, scale_val):
    width = int(input_image.shape[1] * scale_val)
    height = int(input_image.shape[0] * scale_val)

    dim = (width, height)
    result = cv.resize(input_image, dim, interpolation = cv.INTER_AREA)

    return result


def detect_fruits(img_path: str) -> Dict[str, int]:

    image = cv.imread(img_path, cv.IMREAD_COLOR)

    orange_counter = 0
    banana_counter = 0
    apple_counter = 0
    
    with open('dnn_model/coco.names', 'r') as file:
        classes = file.read().splitlines()
    
    net = cv.dnn.readNetFromDarknet('dnn_model/yolov4.cfg', 'dnn_model/yolov4.weights')
    
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    
    classIds, scores, boxes = model.detect(image, confThreshold=0.6, nmsThreshold=0.5)
    
    for (classId, score, box) in zip(classIds, scores, boxes):
        fruit_name = '%s' % (classes[classId])

        if fruit_name == "orange":
            cv.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(0,165,255), thickness=5)
            cv.putText(image, fruit_name, (box[0], box[1]+box[3]-20), cv.FONT_HERSHEY_SIMPLEX, 2,
                    color=(0, 0, 0), thickness=5)
            orange_counter = orange_counter + 1
        
        if fruit_name == "apple":
            cv.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(0,0,255), thickness=5)
            cv.putText(image, fruit_name, (box[0], box[1]+box[3]-20), cv.FONT_HERSHEY_SIMPLEX, 2,
                    color=(0, 0, 0), thickness=5)
            apple_counter = apple_counter + 1

        if fruit_name == "banana":
            cv.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(0,255,255), thickness=5)
            cv.putText(image, fruit_name, (box[0], box[1]+box[3]-20), cv.FONT_HERSHEY_SIMPLEX, 2,
                    color=(0, 0, 0), thickness=5)
            banana_counter = banana_counter + 1

    out_image = ScaleImage(image, 0.3)
    
    banana = str(banana_counter)
    apple = str(apple_counter)
    orange = str(orange_counter)
    
    cv.putText(out_image, ("Bananas: " + banana), (out_image.shape[1]-100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=(0, 0, 0), thickness=1)
    cv.putText(out_image, ("Apples: " + apple), (out_image.shape[1]-100, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=(0, 0, 0), thickness=1)
    cv.putText(out_image, ("Oranges: " + orange), (out_image.shape[1]-100, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=(0, 0, 0), thickness=1)

    # cv.imshow('Fruits Recognison', out_image)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()