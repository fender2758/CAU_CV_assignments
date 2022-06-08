import cv2 as cv
import numpy as np


cap = cv.VideoCapture('That\'s Hilarious.mp4')
cap.set(1, 200)

ret, image = cap.read()

image = image[200:500, 400:]
print(image.shape)

#image = cv.imread("fret_test.png")

Cpy = image.copy()
GRAY = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#edges = cv.Sobel(GRAY, cv.CV_64F, 1, 0, ksize=5)
#edges = cv.convertScaleAbs(edges)

edges = cv.Canny(GRAY,30,30,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/2,50)
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
            cv.line(Cpy,(x1,y1),(x2,y2),(0,0,255),2)
        
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
cv.imshow("original", image)
cv.imshow("image", Cpy)
cv.imshow("canny", edges)
#cv.imshow("cut", image[int(minx):int(maxx)][int(miny):int(maxy)])
#cv.imshow("hough", lines)
cv.waitKey(0)
cv.destroyAllWindows()
