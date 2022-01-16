import cv2 as cv
 
def ScaleImage(input_image, scale_val):
    width = int(input_image.shape[1] * scale_val)
    height = int(input_image.shape[0] * scale_val)

    dim = (width, height)
    result = cv.resize(input_image, dim, interpolation = cv.INTER_AREA)

    return result

def main():
    image = cv.imread('data/01.jpg')
    
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
