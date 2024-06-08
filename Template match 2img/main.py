import cv2 as cv
import numpy as np
import os

haystack = cv.imread('image/Temmies.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle = cv.imread('image/Temmie.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle_w = needle.shape[1]
needle_h = needle.shape[0]
result = cv.matchTemplate(haystack, needle, cv.TM_CCOEFF_NORMED)   



threshold = 0.33
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1])) #Reverse to x,y format, break to two lists and zip

#group overlap rectangles

#List of rectangles[x, y, w, h]
rectangles = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
    rectangles.append(rect)
    rectangles.append(rect) #twice for reducing grouping lose

#print(rectangles)
rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5) # (rectangle_list, the severeness of overlapping to group (usually 1), how close the rectangles to group)

if len(rectangles):
    print(f"{len(locations)} Needle found")
    print(f"{len(rectangles)} Needle found (POST GROUPING)")
    line_color = (0,255,255)
    line_type = cv.LINE_4

    for (x, y, w, h) in rectangles:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv.rectangle(haystack, top_left, bottom_right, line_color, line_type)


else:
    print("No needle found")


cv.imshow("Result", haystack)
cv.waitKey()
cv.destroyAllWindows()