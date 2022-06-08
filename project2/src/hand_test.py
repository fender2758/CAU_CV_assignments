import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
y, sr = librosa.load("cache/audio.wav")



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv.VideoCapture('That\'s Hilarious.mp4')

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) # 또는 cap.get(3)
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) # 또는 cap.get(4)
fps = int(cap.get(cv.CAP_PROP_FPS)) # 또는 cap.get(5)
fps_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))
fourcc = cv.VideoWriter_fourcc(*'DIVX') # 코덱 정의
out = cv.VideoWriter('cache/01.avi', fourcc, 1, (int(width), int(height))) # VideoWriter 객체
out_sound = cv.VideoWriter('cache/02.avi', fourcc, 1, (int(640), int(480))) # VideoWriter 객체

i=0

perc=0

with mp_hands.Hands(
    #model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    for current_frame in range(0, fps_count, 10):
        cap.set(1, current_frame)

        ret, image = cap.read()
        if not ret:
            print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
            break
        """
        print("[", end="")
        for k in range(int(perc)):
            print("=", end="")
        for k in range(33-int(perc)):
            print(".", end="")
        print("]", end="")
        print(str(current_frame)+"/"+str(fps_count), end="")

        print('\r', end='')
        """
        print(str(perc)+"/ 100", end="")

        print('\r', end='')
        
        if int(current_frame*33/fps_count)>=perc:
            perc+=1

        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.

        fig,ax = plt.subplots()
        
        chroma_stft = librosa.feature.chroma_stft(y=y[int(sr*(current_frame/fps)):int(sr*(current_frame/fps+1))], sr=sr)

        
        librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
        
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')

        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        print(img.shape)
        img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
        
        
        fig.clf()
        ax.cla() 
        plt.close(fig)
        print(image.shape)
        cv.imshow("hand", image)
        cv.imshow("chroma", img)
        out.write(image)
        out_sound.write(img)
        if cv.waitKey(5) & 0xFF == 27:
          break
    
    print("\n end")

cap.release()
out.release()
out_sound.release()
