import cv2
import numpy as np
image = cv2.imread("hyonj.png")

height = image.shape[0]
width = image.shape[1]
pt = np.float32([[0,height/2],[0,height],[width-1,0],[width-1,height/2]])
pts2 = np.float32([[0,0],[0,100],[1000,0],[1000,100]])
M = cv2.getPerspectiveTransform(pt, pts2)
for p in pt:
    print(p, M.dot(np.concatenate((p, [1]))))
print("0-000")
fret = cv2.warpPerspective(image, M, (1000,100))
cv2.imshow("1", image)
cv2.imshow("2", fret)
