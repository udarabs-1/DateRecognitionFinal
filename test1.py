import cv2

cap = cv2.VideoCapture(0)
try:
    while True:
        ret, img = cap.read()
        cv2.imshow('webcam', img)
        k = cv2.waitKey(10)
        if k == 27:
            break
except:
    print("Video has ended.")
cap.release()
cv2.destroyAllWindows()
