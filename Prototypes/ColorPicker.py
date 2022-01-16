import cv2 as cv
import numpy as np

def empty():
    pass

def ScaleImage(input_img, scale_val):
    width = int(input_img.shape[1] * scale_val)
    height = int(input_img.shape[0] * scale_val)

    dim = (width, height)
    output_img = cv.resize(input_img, dim, interpolation = cv.INTER_AREA)

    return output_img

cv.namedWindow("HSV")
cv.resizeWindow("HSV", 640, 240)
cv.createTrackbar("B_MIN", "HSV", 0, 255, empty)
cv.createTrackbar("G_MIN", "HSV", 0, 255, empty)
cv.createTrackbar("R_MIN", "HSV", 0, 255, empty)
cv.createTrackbar("B_MAX", "HSV", 255, 255, empty)
cv.createTrackbar("G_MAX", "HSV", 255, 255, empty)
cv.createTrackbar("R_MAX", "HSV", 255, 255, empty)

while True:
    im =  cv.imread('/home/el_zlociako/Documents/WDPO_PROJ/data/01.jpg')

    img = ScaleImage(im, 0.2)

    img_b = cv.GaussianBlur(img, (5, 5), 2)

    imgHSV = cv.cvtColor(img_b, cv.COLOR_BGR2HSV)

    B_min = cv.getTrackbarPos("B_MIN", "HSV")
    B_max = cv.getTrackbarPos("B_MAX", "HSV")
    G_min = cv.getTrackbarPos("G_MIN", "HSV")
    G_max = cv.getTrackbarPos("G_MAX", "HSV")
    R_min = cv.getTrackbarPos("R_MIN", "HSV")
    R_max = cv.getTrackbarPos("R_MAX", "HSV")

    # lower = np.array([B_min, G_min, R_min]) #10 124 122
    # upper = np.array([B_max, G_max, R_max]) #39 255 255

    lower = np.array([0, 90, 0]) #10 124 122
    upper = np.array([255, 255, 255]) #39 255 255
    mask = cv.inRange(imgHSV, lower, upper)

    kernel1 = np.ones((5,5),np.uint8)
    kernel2 = np.ones((31,31),np.uint8)
    dil = cv.dilate(mask,kernel1,iterations = 1)
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel2)

    dupa = dil + closing

    result1 = cv.bitwise_and(img_b, img_b, mask=mask)
    result2 = cv.bitwise_and(img_b, img_b, mask=dupa)
    
    # imgray = cv.cvtColor(img_b, cv.COLOR_BGR2GRAY)
    # contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # chuj = cv.drawContours(img, contours, -1, (0,255,0), 3)

    cv.imshow('Original Image', img)
    cv.imshow('HSV', imgHSV)
    cv.imshow('Color Mask', mask)
    cv.imshow('Res1', result1)
    cv.imshow('Res2', result2)
    # cv.imshow('Result AND', chuj)


    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()



