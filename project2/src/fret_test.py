import cv2
import numpy as np
#from strings import Strings

cap = cv2.VideoCapture('That\'s Hilarious.mp4')
cap.set(1, 200)

ret, image = cap.read()

image = image[200:500, 400:]
image = cv2.imread("../guitar-fingering-recognition-86172fff95313f403475d913c69e564dcd0e0035/pictures2/chordD_2.jpg")
#cv2.imshow("hough", image)
#cv2.waitKey(0)
height = len(image)
width = len(image[0])
image = cv2.resize(image, dsize=(int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)
print(image.shape)
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
#edges = cv2.GaussianBlur(edges, (0, 0), )
#print(edges)
# edges = cv2.medianBlur(edges, 3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, 20, 10)

size = len(lines)

print(size)

for x in range(size):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(neck_with_frets, (x1, y1), (x2, y2), (255, 255, 255), 2)

cv2.imshow("iamge", image)
cv2.imshow("edges", edges)
cv2.imshow("neck_with_frets", neck_with_frets)
cv2.waitKey(0)

neck_str_gray = cv2.cvtColor(neck_with_frets, cv2.COLOR_BGR2GRAY)
# 2. Slice image vertically at different points and calculate gaps between strings at these slices
slices = {}
nb_slices = int(width / 50)
for i in range(nb_slices):
    slices[(i + 1) * 50] = []  # slices dict is {x_pixel_of_slice : [y_pixels_where_line_detected]}
#print(nb_slices)
#print(slices.keys())
for index_line, line in enumerate(neck_str_gray):
    for index_pixel in range(50, len(line), 50):
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
print(slices_differences)
points = []
points_dict = {}
for j in slices_differences.keys():
    gaps = [g for g in slices_differences[j] if g > 1]
    points_dict[j] = []

    if len(gaps) > 3:
        median_gap = np.median(gaps)
        for index, diff in enumerate(slices_differences[j]):
            if abs(diff - median_gap) < 4:
                points_dict[j].append((j, slices[j][index] + int(median_gap / 2)))
            elif abs(diff / 2 - median_gap) < 4:
                points_dict[j].append((j, slices[j][index] + int(median_gap / 2)))
                points_dict[j].append((j, slices[j][index] + int(3 * median_gap / 2)))

    points.extend(points_dict[j])

'''for p in points:
    print(p)
    cv2.circle(neck.image, p, 3, (0, 255, 0), -1)
plt.imshow(cv2.cvtColor(neck.image, cv2.COLOR_BGR2RGB))
plt.show()'''

points_divided = [[] for i in range(5)]
for s in points_dict.keys():
    for i in range(5):
        try:
            # cv2.circle(neck.image, points_dict[s][i], 3, (255, 0, 0), -1)
            points_divided[i].append(points_dict[s][i])
        except IndexError:
            pass

# 3. Use fitLine function to form lines separating each string

tuning = ["E", "A", "D", "G", "B", "E6"]
#strings = Strings(tuning)

for i in range(5):
    cnt = np.array(points_divided[i])
    print(cnt)
    if len(cnt)==0:
        continue
    print(cnt)
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L12, 0, 0.01, 0.01)  # best distType found was DIST_L12

    left_extreme = int((-x * vy / vx) + y)
    right_extreme = int(((width - x) * vy / vx) + y)

    #strings.separating_lines[tuning[i]] = [(width - 1, right_extreme), (0, left_extreme)]

    cv2.line(image, (width - 1, right_extreme), (0, left_extreme), (0, 0, 255), 2)


#print(minx, maxx, miny, maxy)
cv2.imshow("original", image)
cv2.imshow("image", neck_with_frets)
#cv.imshow("canny", edges)
#cv.imshow("cut", image[int(minx):int(maxx)][int(miny):int(maxy)])
#cv.imshow("hough", lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
