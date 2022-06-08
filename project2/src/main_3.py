import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import timeit
from detection.fretdetector import fretDetector

print("Hand detection model..")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

print("VIdeo Capture..")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture("videos/bombombom.mp4")
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
cap.set(6, codec)
cap.set(5, 30)
cap.set(3, 1080)
cap.set(4, 720)
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps=cap.get(cv2.CAP_PROP_FPS)

print("Start")
print("shape = ", width, height)
print("fps = ", fps)
pnt_buffer = np.full((int(fps*3), 4, 2), -1, dtype=np.float32)

win_name = "set_fret"
fretboard = np.zeros((0, 0, 3))
result = np.zeros((0, 0, 3))
search=False

def onMouse(event, x, y, flags, param):
    global pts_cnt, search, result
    if event == cv2.EVENT_LBUTTONDOWN:
        # 좌표에 초록색 동그라미 표시

        # 마우스 좌표 저장
        if pts_cnt < 4:
            cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
            pts[pts_cnt] = [x, y]
            pts_cnt += 1
        cv2.imshow(win_name, draw)
        
        if pts_cnt == 4:
            # 좌표 4개 중 상하좌우 찾기
            sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
            height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            print("shape:", fretboard.shape)
            print("shape:", mtrx.shape)
            print("shape:", (width, height))
            result = cv2.warpPerspective(fretboard, mtrx, (int(width), int(height)))
            search=True
            cv2.imshow('scanned', result)

while(True):
    start_t = timeit.default_timer()
    ret, fretboard = cap.read()
    cv2.imshow("set_fret", fretboard)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cv2.imshow("set_fret", fretboard)
rows, cols = fretboard.shape[:2]
draw = fretboard.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)
cv2.setMouseCallback(win_name, onMouse)
while True:
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
with mp_hands.Hands(
    #model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while(True):
        if search:
            start_t = timeit.default_timer()
            ret, image = cap.read()
            if cv2.waitKey(5) & 0xFF == 27:
              break
            
            src = image.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hand = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    hand.append([(l.x*width, l.y*height, l.z) for l in landmarks])
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    
            hand = np.array(hand)
            if len(hand)==2:
                #print(hand[0, :, 1], hand[1, :, 1])
                #print(hand[0, :, 1].argmin(), hand[0, :, 1].argmax(), hand[1, :, 1].argmin(), hand[1, :, 1].argmax())
                y1_min, y1_max, y2_min, y2_max = hand[0, :, 1].argmin(), hand[0, :, 1].argmax(), hand[1, :, 1].argmin(), hand[1, :, 1].argmax()
                #print(hand[1][y2_min], hand[0][y1_min])
                
                p1 = [0, int(hand[1][y2_min][1] - (hand[1][y2_min][1] - hand[0][y1_min][1])*hand[1][y2_min][0]/(hand[1][y2_min][0] - hand[0][y1_min][0]))]
                p2 = [int(width-1), int(hand[0][y1_min][1] + (hand[1][y2_min][1] - hand[0][y1_min][1])*(width-1 - hand[0][y1_min][0])/(hand[1][y2_min][0] - hand[0][y1_min][0]))]
                
                p3 = [0, int(hand[1][y2_max][1] - (hand[1][y2_max][1] - hand[0][y1_max][1])*hand[1][y2_max][0]/(hand[1][y2_max][0] - hand[0][y1_max][0]))]
                p4 = [int(width-1), int(hand[0][y1_max][1] + (hand[1][y2_max][1] - hand[0][y1_max][1])*(width-1 - hand[0][y1_max][0])/(hand[1][y2_max][0] - hand[0][y1_max][0]))]
                      
                p2[1] = p4[1]+p1[1]-p3[1]

                pts1 = np.float32([p1, p3, p2, p4])
                
                pnt_buffer[:-1] = pnt_buffer[1:]
                pnt_buffer[-1] = pts1
                tab_points = np.array([hand[1, 4], hand[1, 8], hand[1, 12], hand[1, 16], hand[1, 20]])
                print(tab_points)
                #print(tuple(p1), tuple(p2), tuple(p3), tuple(p4))
                
            if not -1 in pnt_buffer:
                # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
                pt = np.median(pnt_buffer, axis=0)
                low  =  int(pt[0][1])if pt[0][1]<pt[2][1] else int(pt[2][1])
                high =  int(pt[1][1])if pt[1][1]>pt[3][1] else int(pt[3][1])
                low = 0 if low<0 else low
                high = height-1 if high>=height else high
                
                fret = src[low:high, :].copy()

                gray1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(fret, cv2.COLOR_BGR2GRAY)

                # ORB, BF-Hamming 로 knnMatch  ---①
                detector = cv2.ORB_create()
                kp1, desc1 = detector.detectAndCompute(gray1, None)
                kp2, desc2 = detector.detectAndCompute(gray2, None)
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(desc1, desc2)

                # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
                matches = sorted(matches, key=lambda x:x.distance)
                # 모든 매칭점 그리기 ---④
                res1 = cv2.drawMatches(result, kp1, fret, kp2, matches, None, \
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

                # 매칭점으로 원근 변환 및 영역 표시 ---⑤
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
                # RANSAC으로 변환 행렬 근사 계산 ---⑥
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h,w = result.shape[:2]
                pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                dst = cv2.perspectiveTransform(pts,mtrx)
                img2 = cv2.polylines(fret,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                # 정상치 매칭만 그리기 ---⑦
                matchesMask = mask.ravel().tolist()
                res2 = cv2.drawMatches(result, kp1, fret, kp2, matches, None, \
                                    matchesMask = matchesMask,
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                # 모든 매칭점과 정상치 비율 ---⑧
                accuracy=float(mask.sum()) / mask.size
                print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
                cv2.imshow('Matching-All', res1)
                cv2.imshow('Matching-Inlier ', res2)
            cv2.imshow("hand", image)

    
#cv2.destroyAllWindows()

cap.release()
