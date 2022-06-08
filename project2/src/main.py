import cv2
from detection.fretdetection import fretDetection

cap = cv2.VideoCapture('anybody there.mp4')
cap.set(1, 2000)

ret, image = cap.read()

cap = cv2.VideoCapture(0)

#fgbg = cv2.createBackgroundSubtractorMOG2()
while(True):
    ret, image = cap.read()
    if ret:
        #fgmask = fgbg.apply(image)
        cv2.imshow('original', image)
        #cv2.imshow('bgsub',fgmask)
        #cv2.waitKey()
        #image = image[:500, :]

        #image = cv2.imread("../guitar-fingering-recognition-86172fff95313f403475d913c69e564dcd0e0035/pictures2/chordAm.jpg")
        #image = cv2.imread("chordD_2.jpg")
        try:
            img, fret, edges = fretDetection(image, 50)

            cv2.imshow("lined", img)
            cv2.imshow("edges", edges)
            cv2.imshow("fret", fret)
        except:
            continue
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
print("end")
cap.release()
cv2.destroyAllWindows()
