import cv2 as cv
import numpy as np

def ScaleImage(input_img, scale_val):
    width = int(input_img.shape[1] * scale_val)
    height = int(input_img.shape[0] * scale_val)

    dim = (width, height)
    output_img = cv.resize(input_img, dim, interpolation = cv.INTER_AREA)

    return output_img

def main():
    img1 = cv.imread('/home/el_zlociako/Documents/WDPO_PROJ/testing/banana1.png', 1)
    img2 = cv.imread('/home/el_zlociako/Documents/WDPO_PROJ/data/02.jpg', 1)

    resized1 = ScaleImage(img1, 0.2)
    resized2 = ScaleImage(img2, 0.2)

    gray1 = cv.cvtColor(resized1 ,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(resized2 ,cv.COLOR_BGR2GRAY)


    # gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)

    # sobel = cv.Sobel(gray,cv.CV_8U,1,1,ksize=5)

    # blur = cv.GaussianBlur(gray, (5,5), 0)

    # edges = cv.Canny(blur,0,200)

    # contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # chuj = cv.drawContours(resized, contours, -1, (0,255,0), 3)
    # img2gray = cv.cvtColor(img_resized,cv.COLOR_BGR2GRAY)
    # img2gray = np.float32(img2gray)
    # img_blur = cv.medianBlur(img2gray, 3)

    # ret, mask = cv.threshold(img2gray, 120, 150, cv.THRESH_BINARY)
    # mask_inv = cv.bitwise_not(mask)
    
    # dst = cv.cornerHarris(mask_inv,2,3,0.04)
    # img_resized[dst>0.01*dst.max()]=[255,0,0]

    # cv.imshow('1', sobel)
    # cv.imshow('2', edges)
    cv.imshow('1', imgKp1)
    cv.imshow('2', imgKp2)
    cv.imshow('3', img3)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()  

