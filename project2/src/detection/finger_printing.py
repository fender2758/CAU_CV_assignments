import cv2
import numpy as np

first_fret = 100
print_frets = [0, first_fret]
for i in range(9):
    print_frets.append(int((print_frets[-1]-print_frets[-2])*16.817/17.817+print_frets[-1]))


font =  cv2.FONT_HERSHEY_PLAIN
def print_fingering(fingers):
    board = np.zeros((100, print_frets[-1]))
    for i in range(11):
        for f in print_frets:
            cv2.line(board, (int(f), 0), (int(f), 99), (255, 255, 255), 2)
    for i in range(6):
        for f in print_frets:
            cv2.line(board, (5, 5+18*i), (print_frets[-1], 5+18*i), (255, 255, 255), 2)
    for i, (l, f) in enumerate(fingers):
        if f!=0:
            fr = int((print_frets[f]+print_frets[f-1])/2)
            li = 5+18*(5-l)
            cv2.circle(board, (fr, li), 5, (0,0,0),-1)
            board = cv2.putText(board, str(i), (fr-5, li+4), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("printed_board", board)

a = [[-1, 0], [4, 1], [2, 2], [1, 3], [-1, 0]]

print_fingering(a)
