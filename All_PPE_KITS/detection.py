import math
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0) # if you have multiple webcam then you change the value of 0 to 1
#opening the video file using opencv's videocapture class
frame_width = int(cap.get(3)) # getting the frame width and height
frame_height = int(cap.get(4)) #Get the width and height of each frame using cap.get().

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))
#After detection output video will save in output.avi file
# 10 is time frame rate

model=YOLO("../YOLO-Weights/detection.pt") # passing the path of algorithm
classNames = ['boots', 'gloves', 'goggles', 'hardhat', 'mask', 'no boots', 'no gloves', 'no goggles', 'no hardhat', 'no mask', 'no suit', 'suit']

#The COCO (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset.
# models trained on COCO, which include 80 pre-trained classes.



while True:
    success,img = cap.read()
    # used for detection using YOLOv8 frame by frame
    # stream = True will use the generator and it is more efficient than normal
    results = model(img,stream=True)
    # result of video will store in  the results variable
    #once we get the result we can check individual bounding boxes and see how well it performs
    #Then we  will loop through them and we will have bounding boxes for each of the results
    # we will loop through each of the bounding box
    cv2.imshow("Image",img)
    #Display the original frame using cv2.imshow().
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            # here x1,y1 is a top-left corner co-ordinates of bounding box ans x2,y2 is a bottom-right corner co-ordinates
            #print(x1,y1,x2,y2)
            x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
            #we convert this co-ordinates in the integer value otherwise it gives output in the form of tensor
            # for detection we need output in the form of integer
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            # we have the co-ordinates values so, using cv2 we create the rectangle around  each of the detecting object
            # we have to pass cv2.rectangle(image,start_point,end_point,color,thickness)

            #print(box.conf[0])

            conf = math.ceil((box.conf[0]*100))/100  # it gives confidence score value  and it convert into integer because it is in the form of tensor
            cls = int(box.cls[0])   #getting the class-id  (0-person,1-bicycle)
            class_name = classNames[cls]   #gives the class name

            label = f'{class_name}{conf}'   #gives the class and confidence value combine
            t_size = cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]     #finding the size of the rectangle
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img,(x1,y1),c2,[255,0,255],-1,cv2.LINE_AA)
            cv2.putText(img,label,(x1,y1-2),0,1,[255,255,255],thickness=1,lineType=cv2.LINE_AA)
        out.write(img)      #save output in output file
        cv2.imshow("Image",img)


    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
out.release()