import cv2 as cv2
import numpy as np


image = cv2.imread("../guitar-fingering-recognition-86172fff95313f403475d913c69e564dcd0e0035/pictures2/chordD_2.jpg")
#cv2.imshow("hough", image)
#cv2.waitKey(0)
print(image.shape)
height = len(image)
width = len(image[0])
image = cv2.resize(image, dsize=(int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)
Cpy = image.copy()
height = len(image)
width = len(image[0])
neck_with_frets = np.zeros((height, width, 3), np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 1. Detect frets with Hough transform and form an Image based on these

edges = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
#print(edges.shape)


kernel = np.ones((11, 11), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
red, edges = cv2.threshold(edges,127,255, cv2.THRESH_BINARY)
lines = cv2.HoughLines(edges,1,np.pi/180,150)
num_lines = lines.shape[0]

print(lines.shape)
sq_lines = np.reshape(np.tile(lines, reps=(num_lines, 1)), (num_lines, num_lines, 2))
print(sq_lines.shape)
print(sq_lines[0])
sq_lines = sq_lines-sq_lines.swapaxes(0, 1)
#print(sq_lines.shape)
#print((sq_lines-sq_lines_T)[0])


#ver_lines = cv.HoughLines(edges,1,np.pi/180,50)
print(lines.shape)

rho,theta = lines[0][0]

a = np.cos(theta)
b = np.sin(theta)
maxx = a*rho
minx = a*rho
maxy = b*rho
miny = b*rho


for i, line in enumerate(lines):
    rho,theta = line[0]
    
    a = np.cos(theta)
    b = np.sin(theta)
    for j, line in enumerate(sq_lines[i]):
        #if abs(abs(line[1])-np.pi/2)<np.pi/(180/4):
        if True:
            
        
            x0 = a*rho
            y0 = b*rho  
            
                
            x1 = int(x0 + 1000*(-b))
            x2 = int(x0 - 1000*(-b))
            y1 = int(y0 + 1000*(a))
            y2 = int(y0 - 1000*(a))
            cv2.line(Cpy,(x1,y1),(x2,y2),(0,0,255),2)
        
    if y0>maxy:
        maxy = y0
    if y0<miny and y0!=0:
        miny = y0
        
    """x1 = int(x0 + 2000*(-b))
    x2 = int(x0 - 2000*(-b))
    y1 = int(y0 + 2000*(a))
    y2 = int(y0 - 2000*(a))
    cv.line(Cpy,(x1,y1),(x2,y2),(0,0,255),2)"""

"""
for line in ver_lines:
    rho,theta = line[0]
    
    a = np.cos(theta)
    b = np.sin(theta)
    
    x0 = a*rho
    y0 = b*rho
    
    if np.pi/(180/3)>theta>-1*(np.pi/(180/3)):
        if x0>maxx:
            maxx = x0
        if x0<minx and x0>0:
            minx = x0

            
        x1 = int(x0 + 100*(-b))
        x2 = int(x0 - 100*(-b))
        y1 = int(y0 + 200*(a))
        y2 = int(y0 + 0*(a))
        cv.line(Cpy,(x1,y1),(x2,y2),(0,0,255),2)
"""
print(minx, maxx, miny, maxy)
cv2.imshow("original", image)
cv2.imshow("image", Cpy)
cv2.imshow("canny", edges)
#cv.imshow("cut", image[int(minx):int(maxx)][int(miny):int(maxy)])
#cv.imshow("hough", lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
