import cv2 as cv
import numpy as np
import os


haystack = cv.imread('image/Temmies.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle = cv.imread('image/Temmie.jpg', cv.IMREAD_REDUCED_COLOR_2)


result = cv.matchTemplate(haystack, needle, cv.TM_CCOEFF_NORMED)

#get best match pos
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

#match threshold
threshold = 0.8

if max_val >= threshold:
    needle_w = needle.shape[1]
    needle_h = needle.shape[0]
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    cv.rectangle(haystack, top_left, bottom_right, color=(0,0,255), thickness=2, lineType=cv.LINE_4)
else:
    print("Not found")


#Show image
# cv.imshow('result', haystack)
# cv.waitKey()
# cv.destroyAllWindows()

#Save result image as file
#cv.imwrite('image/result.jpg', haystack)