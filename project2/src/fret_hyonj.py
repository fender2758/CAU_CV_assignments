import cv2
import numpy as np
#from strings import Strings

cap = cv2.VideoCapture('That\'s Hilarious.mp4')
cap.set(1, 200)

ret, image = cap.read()

image = image[200:500, 400:]
image = cv2.imread("../guitar-fingering-recognition-86172fff95313f403475d913c69e564dcd0e0035/pictures2/chordAm.jpg")
#cv2.imshow("hough", image)
#cv2.waitKey(0)
height = len(image)
width = len(image[0])
if height>1440:
    hei=1440
    wid=int(1440*width/height)
    image = cv2.resize(image, dsize=(wid, hei), interpolation=cv2.INTER_AREA)
    height, width = hei, wid
    #image = cv2.resize(image, dsize=(int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)
base = image.copy()
print(image.shape)
height = len(image)
width = len(image[0])
neck_with_frets = np.zeros((height, width, 3), np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 1. Detect frets with Hough transform and form an Image based on these

edges = cv2.Sobel(gray, cv2.CV_8U, 0, 2, 3)
#print(edges.shape)


kernel = np.ones((11, 11), np.uint8)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
red, edges = cv2.threshold(edges,50,255, cv2.THRESH_BINARY)
#edges = cv2.GaussianBlur(edges, (0, 0), )
#print(edges)
# edges = cv2.medianBlur(edges, 3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, 30, 5)

size = len(lines)

#print(size)

for x in range(size):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(neck_with_frets, (x1, y1), (x2, y2), (255, 255, 255), 2)


    neck_str_gray = cv2.cvtColor(neck_with_frets, cv2.COLOR_BGR2GRAY)
# 2. Slice image vertically at different points and calculate gaps between strings at these slices
slices = {}
split = 10
nb_slices = int(width / split)
for i in range(nb_slices):
    slices[(i + 1) * split] = []  # slices dict is {x_pixel_of_slice : [y_pixels_where_line_detected]}
#print(nb_slices)
#print(slices.keys())
#print(len(neck_str_gray[0]))
for index_line, line in enumerate(neck_str_gray):
    for index_pixel in range(split, len(line), split):
        #print(pixel)
        if line[index_pixel] == 255:
            slices[index_pixel].append(index_line)
            
#print(slices)
slices_differences = {}  # slices_differences dict is {x_pixel_of_slice : [gaps_between_detected_lines]}
for k in slices.keys():
    temp = []
    n = 0
    slices[k] = list(sorted(slices[k]))
    for p in range(len(slices[k]) - 1):
        temp.append(slices[k][p + 1] - slices[k][p])
        if slices[k][p + 1] - slices[k][p] > 1:
            n += 1
    slices_differences[k] = temp
    
#print(slices_differences)
points = []
points_dict = {}
Cpy = image.copy()
for j in slices_differences.keys():
    up_bound = len(slices_differences[j])-1
    low_bound = 0
    gaps = []
    
    gaps = [(i, g) for i, g in enumerate(slices_differences[j]) if g > 10 and g <30]
    
    gaps = sorted(gaps, key=lambda x : x[1])
    #while gaps[:-1] 
    points_dict[j] = []
    temp = []

    if len(gaps) > 3:
        median_gap = np.median(gaps)
        #print(median_gap)
        for index, diff in enumerate(slices_differences[j]):
            #print(diff, end=" ")
            if abs(diff - median_gap) < median_gap/4:
                #print(diff, median_gap, (j, slices[j][index] + int(median_gap / 2)))
                temp.append((j, slices[j][index] + int(median_gap / 2)))
                cv2.circle(Cpy, (j, slices[j][index] + int(median_gap / 2)), 3, (0, 0, 255), -1)
            elif abs(diff / 2 - median_gap) < median_gap/8:
                temp.append((j, slices[j][index] + int(median_gap / 2)))
                temp.append((j, slices[j][index] + int(3 * median_gap / 2)))
                cv2.circle(Cpy, (j, slices[j][index] + int(median_gap / 2)), 3, (0, 0, 255), -1)
                cv2.circle(Cpy, (j, slices[j][index] + int(3*median_gap / 2)), 3, (0, 0, 255), -1)
        #print(temp)
        for i, t in enumerate(temp[:-1]):
            #print(t,abs(temp[i+1][1]-temp[i][1]-median_gap), abs(temp[i+1][1]-temp[i][1]-median_gap)<median_gap/2)
            if abs(temp[i+1][1]-temp[i][1]-median_gap) < median_gap/2:
                points_dict[j].append(t)
                if i==len(temp)-2:
                    #print(temp[i+1],abs(temp[i+1][1]-temp[i][1]-median_gap), abs(temp[i+1][1]-temp[i][1]-median_gap)<median_gap/2)
                    points_dict[j].append(temp[i+1])
        cv2.imshow("test_dor", Cpy)
        #cv2.imshow("image", neck_with_frets)
        #cv2.waitKey(0)

        #print(points_dict[j])
                    
                
    points.extend(points_dict[j])


'''for p in points:
    print(p)
    cv2.circle(neck.image, p, 3, (0, 255, 0), -1)
plt.imshow(cv2.cvtColor(neck.image, cv2.COLOR_BGR2RGB))
plt.show()'''

tun_color = [(255, 0, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (0, 255, 255)]
points_divided = [[] for i in range(5)]

decs = np.array([p[-1][1]-p[0][1] for p in points_dict.values() if len(p)!=0])
bins=[i for i in range((np.min(decs)//10)*10, (np.max(decs)//10+1)*10, 10)]
d = np.digitize(decs, bins=bins)
decs = sorted(decs)
median_dec = np.median(decs)
cv2.waitKey(0)
for s in points_dict.keys():
    if len(points_dict[s])!=0 and abs(points_dict[s][-1][1]-points_dict[s][0][1]-median_dec) < median_dec/10:
        continue
    
    for i in range(len(points_dict[s])):
        try:
            #print(i//2 if i%2==0 else len(points_dict[s])-i//2-1, i//2 if i%2==0 else 5-i//2-1)
            index = points_dict[s][i//2 if i%2==0 else len(points_dict[s])-i//2-1]
            cv2.circle(image, points_dict[s][i//2 if i%2==0 else len(points_dict[s])-i//2-1], 3, tun_color[i//2 if i%2==0 else 5-i//2-1], -1)
            points_divided[i//2 if i%2==0 else 5-i//2-1].append(points_dict[s][i//2 if i%2==0 else len(points_dict[s])-i//2-1])
        except IndexError:
            pass

# 3. Use fitLine function to form lines separating each string

tuning = ["E", "A", "D", "G", "B", "E6"]
#strings = Strings(tuning)
x1 = 0
y1_min = height
y1_max = 0
x2 = width-1
y2_min = height
y2_max = 0
for i in range(5):
    cnt = np.array(points_divided[i])
    #print(cnt)
    if len(cnt)==0:
        continue
    #print(cnt)
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L12, 0, 0.01, 0.01)  # best distType found was DIST_L12

    left_extreme = int((-x * vy / vx) + y)
    right_extreme = int(((width - x) * vy / vx) + y)

    #strings.separating_lines[tuning[i]] = [(width - 1, right_extreme), (0, left_extreme)]
    if left_extreme<y1_min:
        y1_min = left_extreme
    if left_extreme>y1_max:
        y1_max = left_extreme
    if right_extreme<y2_min:
        y2_min = right_extreme
    if right_extreme>y2_max:
        y2_max = right_extreme
    cv2.line(image, (width - 1 - i*10, right_extreme), (0, left_extreme), tun_color[i], 2)

margin1=int((y1_max-y1_min)/4)
margin2=int((y2_max-y2_min)/4)
#cv2.imshow("fret_npmargin", image[x1:x2][y1:y2])
y1_min-=margin1
y1_max+=margin1
y2_min-=margin2
y2_max+=margin2
#print(y1_min, y1_max)
#print(y2_min, y2_max)

# 좌표의 이동점
pts1 = np.float32([[0,y1_min],[0,y1_max],[x2,y2_min],[x2,y2_max]])
pts2 = np.float32([[0,0],[0,100],[1000,0],[1000,100]])

# pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
#cv2.circle(image, (0,y1_min), 20, (255,0,0),-1)
#cv2.circle(image, (0,y1_max), 20, (0,255,0),-1)
#cv2.circle(image, (x2,y2_min), 20, (0,0,255),-1)
#cv2.circle(image, (x2,y2_max), 20, (0,0,0),-1)

M = cv2.getPerspectiveTransform(pts1, pts2)
fret = cv2.warpPerspective(base, M, (1000,100))
#for k in slices.keys():
#    cv2.line(image, (0, k), (width, k), (0, 0, 255), 2)


#print(minx, maxx, miny, maxy)
cv2.imshow("original", image)
cv2.imshow("fret", fret)
cv2.imshow("image", neck_with_frets)
#cv2.imshow("fret", image[x1:x2][y1:y2])
#cv.imshow("canny", edges)
#cv.imshow("cut", image[int(minx):int(maxx)][int(miny):int(maxy)])
#cv.imshow("hough", lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
