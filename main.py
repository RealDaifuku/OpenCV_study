import cv2 as cv
import numpy as np

haystack_img = cv.imread('Temmies.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('Temmie.jpg', cv.IMREAD_UNCHANGED)

if haystack_img is None or needle_img is None:
    print("Error: One or both images not loaded.")

else:
    result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    width = needle_img.shape[1]
    height = needle_img.shape[0]

    threshold = 0.80
    yloc, xloc = np.where(result >= threshold)
    
    rectangles = []
    for (x, y) in zip(xloc, yloc):
        rectangles.append([int(x), int(y), int(width), int(height)])
        rectangles.append([int(x), int(y), int(width), int(height)])

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.2)

    print("Matched", len(rectangles))

    for (x,y,w,h) in rectangles:
        cv.rectangle(haystack_img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv.imshow('Result', haystack_img)
    cv.waitKey()
    cv.destroyAllWindows()