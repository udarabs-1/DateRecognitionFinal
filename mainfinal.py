import cv2
import numpy as np
from segmentation import get_yolov5, get_image_from_bytes
from easyocr import Reader
import time
import pandas as pd
import io
from PIL import Image



cap = cv2.VideoCapture(0)

cap.set(3, 416)
cap.set(4, 416)
model = get_yolov5()
reader = Reader(['en'], gpu = True)

  
dim = (416,416)


def drawRectangles(image, dfResults):
    for index, row in dfResults.iterrows():
        print((row['xmin'], row['ymin']))
        image = cv2.rectangle(image, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (255, 0, 0), 2)
    cv2.imshow("mm",image)

try:
    while True:
        ts = time.time()
        #ret, frame = cap.read()
        #img = cv2.resize(frame, (416, 416))
        #cv2.imshow('mmm', img)
        frame = Image.open('test1.jpg')
        screen = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        detections = model(frame, size = 416)
        #detections.save("adoo.jpg")
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        print(results)
        i=0
        for result in results:
                    con = result['confidence']
                    cs = result['name']
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    bbox_points=[x1, y1, x2, y2]
                    confidence_score = con
                    object_name = cs

                    print('bounding box is ', x1, y1, x2, y2)
                    print('detected object name is ', object_name)
                    im0 = screen
                
                    cropped_img = im0[y1:y2, x1:x2]
                    grayImage = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    screen = cv2.cvtColor(np.array(grayImage), cv2.COLOR_RGB2BGR)
                    frame1 = cv2.resize(screen, None, fx=0.3, fy=0.3)
                    
                    #print(frame1.shape[0])
                    #print(frame1.shape[1])
                    results = reader.readtext(frame1)
                    #print(results)
            
                    print(results[0][1][0:3]+" "+ results[1][1])
                    print(results[2][1][0:3] + " "+ results[2][1][-10:])
                    print(results[3][1][0:3] + " "+ results[3][1][-12:])
                    print(results[4][1][0:3] + " "+ results[4][1][-6:])
                    te = time.time()
                    td = te - ts
                    print(f'Completed in {td} seconds')
                    print("")
                    print("")
                    print("")
                    i= i+1
except:
    print("Video has ended.")

    #cv2.imshow("mmm", )
    cv2.imwrite("cropped.png", cropped_img)
    #result = model(frame, size = 416)
    #result.print()
    #result.crop()
    #result.save("save.jpg")
    #result.display(render = True)
    #model.save()
    #detections.view_img()
cap.release()
cv2.destroyAllWindows()

