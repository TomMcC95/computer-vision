import cv2 as cv
import numpy as np

image_1 = cv.imread(r"C:\Users\tmccl\Pictures\image_1.jpg")
image_2 = cv.imread(r"C:\Users\tmccl\Pictures\image_2.jpg")

def belt_join_check(image_2, image_1 = belt_join)

    image_1 = cv.cvtColor(image_1, cv.COLOR_BGR2HSV) #hsv_
    image_2 = cv.cvtColor(image_2, cv.COLOR_BGR2HSV) #hsv_

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists

    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_image_1 = cv.calcHist([image_1], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_image_1, hist_image_1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_image_2 = cv.calcHist([image_2], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_image_2, hist_image_2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    one_two = cv.compareHist(hist_image_1, hist_image_2, 0)

    if one_two > 0.6:
        return False

    else:
        return True

#for compare_method in range(4):
#    one_one = cv.compareHist(hist_image_1, hist_image_1, compare_method)
#    one_two = cv.compareHist(hist_image_1, hist_image_2, compare_method)
#    two_two = cv.compareHist(hist_image_2, hist_image_2, compare_method)
#    print('Method:', compare_method, '1_1 / 1_2 / 2_2 :',\
#          one_one, '/', one_two, '/', two_two)