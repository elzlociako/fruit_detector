<<<<<<< HEAD
import cv2 as cv
 
def ScaleImage(input_image, scale_val):
    width = int(input_image.shape[1] * scale_val)
    height = int(input_image.shape[0] * scale_val)

    dim = (width, height)
    result = cv.resize(input_image, dim, interpolation = cv.INTER_AREA)

    return result

def main():
    image = cv.imread('data/03.jpg')
    
    with open('dnn_model/coco.names', 'r') as file:
        classes = file.read().splitlines()
    
    net = cv.dnn.readNetFromDarknet('dnn_model/yolov4.cfg', 'dnn_model/yolov4.weights')
    
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    
    classIds, scores, boxes = model.detect(image, confThreshold=0.6, nmsThreshold=0.4)
    
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(255, 0, 0), thickness=5)
    
        fruit_name = '%s' % (classes[classId])
        cv.putText(image, fruit_name, (box[0], box[1]+box[3]-20), cv.FONT_HERSHEY_SIMPLEX, 2.5,
                    color=(0, 0, 0), thickness=5)
    
    out_image = ScaleImage(image, 0.3)
    cv.imshow('Fruits Recognison', out_image)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()  
=======
import cv2 as cv
import json
import click

from glob import glob
from tqdm import tqdm

from typing import Dict


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    #TODO: Implement detection method.
    
    apple = 0
    banana = 0
    orange = 0

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):

    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits
    
    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
>>>>>>> 91bb03f8f8917e7ffad162bd192277f82c929c50
