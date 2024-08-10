import cv2 as cv
import numpy as np

haystack_img = cv.imread('D:\Project\Code\OpenCVAlbion\Tutorial\Resources\OpenWorld_1_main.png', cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread('D:\Project\Code\OpenCVAlbion\Tutorial\Resources\OpenWorld_1_T3ore.png', cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

#get best match pos
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

#____DEBUG_____
#print('Best match top left pos : %s' % str(max_loc))
#print('Best match confidance: %s' % max_val)
#_____DEBUG_____

threshold = 0.9
if max_val >= threshold:
    print('Found needle')
    
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] +needle_h)
    
    cv.rectangle(haystack_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)
    cv.imshow('Result', haystack_img)
    cv.waitKey()
else:
    print('404')
