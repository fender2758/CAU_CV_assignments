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
codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
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

#pnt_buffer = np.full((2, 4, 2), -1, dtype=np.float32)
pnt_buffer = np.full((int(fps/2), 4, 2), -1, dtype=np.float32)
tab_points= np.zeros((5, 2))
fret = np.zeros((100, 1000, 3))

fret_Detector = fretDetector(fps, 720, 1080)


def get_fingers(hand):
    result =[]
    LR = 1 if hand[1, 8, 0]>hand[0, 8, 0] else 0
    result.append(np.concatenate((hand[LR, 4, :2], [1])))
    result.append(np.concatenate((hand[LR, 8, :2], [1])))
    result.append(np.concatenate((hand[LR, 12, :2], [1])))
    result.append(np.concatenate((hand[LR, 16, :2], [1])))
    result.append(np.concatenate((hand[LR, 20, :2], [1])))
    return np.array(result)

with mp_hands.Hands(
    #model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while(True):
        
        error_message = ""
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
            tab_points = get_fingers(hand)
            #print(tab_points)
            #print(tuple(p1), tuple(p2), tuple(p3), tuple(p4))
        else:
            error_message = "Two hands please"
        if not -1 in pnt_buffer:
            # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
            pt = np.median(pnt_buffer, axis=0)
            low  =  int(pt[0][1])if pt[0][1]<pt[2][1] else int(pt[2][1])
            high =  int(pt[1][1])if pt[1][1]>pt[3][1] else int(pt[3][1])
            low = 0 if low<0 else low
            high = int(height)-1 if high>=height else high
            
            fret = src[low:high, :].copy()
            """
            if -1 not in fret_Detector.fret_buff:
                img, fret__, edges = fret_Detector.fretDetection_board(fret__, 230)
                cv2.imshow("lined", img)
                #cv2.imshow("edges", edges)
                cv2.imshow("fret_", fret__)
            """
            """
            
            img, fret__, edges, M = fret_Detector.fretDetection(src, low, high, 150)
            
            #cv2.imshow("lined", img)
            5
            #cv2.imshow("edges", edges)
            cv2.imshow("fret__", fret__)
            if -1 not in fret_Detector.fret_buff:
                fret_Detector.fretDetection_board(fret__, 254)
                fret_Detector.print_fret()
                
                cv2.imshow("lined", img)
                cv2.imshow("edges", edges)
                #cv2.imshow("fret_", fret_)
                cv2.imshow("fret__", fret__)
            """
            try:
                img, fret__, edges, M = fret_Detector.fretDetection(src, low, high, 150)
                
                if len(hand)==2:
                    for i, tab in enumerate(tab_points):
                        tab_points[i] = M.dot(tab)
                        tab_points[i]/=tab_points[i][2]
                        
                fret_Detector.update_tabs(tab_points)
            
                
                tab_points = tab_points.dot(M)
                    
                #cv2.imshow("lined", img)
                #cv2.imshow("edges", edges)
                #cv2.imshow("fret__", fret__)
                if -1 not in fret_Detector.fret_buff:
                    fret_Detector.fretDetection_board(fret__, 254)
                    fret_Detector.print_fret()
                    fingers = fret_Detector.get_fingering()
                    
                    finger_match = fret_Detector.print_fingering(fingers)
                    #cv2.imshow("lined", img)
                    #cv2.imshow("edges", edges)
                    #cv2.imshow("fret_", fret_)
                    #cv2.imshow("fret__", fret__)
            except:
                finger_match = fret_Detector.print_fingering()
                error_message ="no guitar"
        
        else:
            finger_match = fret_Detector.print_fingering()
                
        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))
        
        font =  cv2.FONT_HERSHEY_PLAIN
        print_ = cv2.flip(image, 1)
        print_ = cv2.putText(print_, error_message, (0, 100), font, 2, (0, 0, 255), 1, cv2.LINE_AA)

        
        finger_match = cv2.resize(finger_match, (print_.shape[1], int(100*print_.shape[1]/finger_match.shape[1])), interpolation = cv2.INTER_LINEAR )
                                  
        finger_match = finger_match.astype(np.uint8)
        
        print_ = cv2.vconcat([print_, finger_match]) # axis = 1 과 결과 동일
        cv2.imshow("print_", print_)
        #cv2.imshow("fret", fret)



cv2.destroyAllWindows()

cap.release()
