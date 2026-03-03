import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)


#Function to draw crosshairs at the middle
def crosshairs(frame):
    height, width, channels = frame.shape

    #Get centre row and centre column

    centre_row = height//2
    centre_column = width//2

    #Draw the two lines to form a crosshair
    
    line_1 = cv.line(frame, ((centre_column-5),(centre_row)),(centre_column+5,centre_row), (0,0,255))
    line_2 = cv.line(frame, ((centre_column),(centre_row-5)),(centre_column,centre_row+5), (0,0,255))

    return frame
    

if not cap.isOpened():
    print("Cannot open the camera")
    exit()

while True:
    ret, frame = cap.read()


    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_to_display = crosshairs(frame)
    cv.imshow('frame', frame_to_display)


    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()