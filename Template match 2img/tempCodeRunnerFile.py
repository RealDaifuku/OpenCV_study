import cv2 as cv
import numpy as np

def match(hay_path, needle_path, threshold = 0.5,method = cv.TM_CCOEFF_NORMED,where = 1):
    haystack = cv.imread(hay_path, cv.IMREAD_REDUCED_COLOR_2)
    needle = cv.imread(needle_path, cv.IMREAD_REDUCED_COLOR_2)
    needle_w = needle.shape[1]
    needle_h = needle.shape[0]
    result = cv.matchTemplate(haystack, needle, method)
    if where:
        locations = np.where(result >= threshold)
    else:
        locations = np.where(result <= threshold)
    locations = list(zip(*locations[::-1]))

    rectanges = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
        rectanges.append(rect)
        rectanges.append(rect)

    rectangles, weights = cv.groupRectangles(rectanges, 1, 0.5)

    if len(rectanges):
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

match('image/Temmies.jpg', 'image/Temmie.jpg',0.33)
