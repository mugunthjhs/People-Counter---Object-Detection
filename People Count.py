from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *

# Open video file
video_path = "people.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

model = YOLO("YOLO Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", 
              "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
              "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", 
              "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
              "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", 
              "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
              "sink", "refrigerator", "book", "clock", "vase", "scissors", 
              "teddy bear", "hair drier", "toothbrush","van"]

mask = cv2.imread("mask.png")

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limitsup = [103, 161, 296, 161]
limitsdown = [527, 489, 735, 489]

totalCountup = []
totalCountdown = []

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
output_video_path = "output_people_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, img = cap.read()
    
    # Exit if video stream has ended or there was an error reading the frame
    if not success:
        print("End of video stream or read error.")
        break  

    # Warning for invalid image
    if img is None or img.size == 0:
        print("Warning: Image is not valid or empty.")
        continue  # Skip this iteration to avoid processing an empty frame
    
    
    imgRegion = cv2.bitwise_and(img,mask)
    
    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(730,260))
    results = model(imgRegion, stream=True)
    
    detections = np.empty((0,5))
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w, h = x2 - x1, y2 - y1
           

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            
            # Class name
            cls = box.cls[0]
            currentClass = classNames[int(cls)]
            

            # Put text rectangle
            if currentClass == "person" and conf > 0.3:
                #cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0, x1), max(35, y1)), thickness=1, scale=1,offset=5)
                #cvzone.cornerRect(img, (x1, y1, w, h),l=10)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    
    
    overlay = img.copy()
    alpha = 0.01 # Set the transparency level
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.line(overlay,(limitsup[0], limitsup[1]),(limitsup[2], limitsup[3]),(0,0,255),5)
    cv2.line(overlay,(limitsdown[0], limitsdown[1]),(limitsdown[2], limitsdown[3]),(0,0,255),5)
    
    for results in resultsTracker:
        x1,y1,x2,y2,id = results
        print(results)
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1-10)), thickness=2, scale=3, offset=10)
        
        cx,cy = x1+w//2,y1+h//2
        overlay = img.copy()
        cv2.circle(overlay,(cx,cy),5,(255,0,255),cv2.FILLED)
        alpha = 0.01 # Set the transparency level
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


        if limitsup[0] < cx < limitsup[2] and limitsup[1]-15 < cy < limitsup[1]+15:
            if totalCountup.count(id) == 0:
                totalCountup.append(id)
                cv2.line(img,(limitsup[0], limitsup[1]),(limitsup[2], limitsup[3]),(0,255,0),5)
            
            
        if limitsdown[0] < cx < limitsdown[2] and limitsdown[1]-15 < cy < limitsdown[1]+15:
            if totalCountdown.count(id) == 0:
                totalCountdown.append(id)
                cv2.line(img,(limitsdown[0], limitsdown[1]),(limitsdown[2], limitsdown[3]),(0,255,0),5)
                
                
    #cvzone.putTextRect(img,f"Total Count :{len(totalCount)}",(50,50))
    cv2.putText(img,str(len(totalCountup)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(totalCountdown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)
    
    out.write(img)
    
    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)

    # Break loop on 'q' key press
    cv2.waitKey(1)
    

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
