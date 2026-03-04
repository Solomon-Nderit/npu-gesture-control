import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open the camera")
    exit()

for i in range(50):
    ret, frame = cap.read()

    cv.imwrite(f"calib_data/image_{i}.jpg", frame)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
