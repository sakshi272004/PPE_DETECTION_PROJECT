from ultralytics import YOLO
# ultralytics is a package from which we directly included YOLO algorithm
#Ultralytics HUB is an intuitive AI platform for creating, training, and deploying machine learning models with a no-code interface and deep learning framework support.

import cv2
#cv2 is the OpenCV library used for computer vision tasks.

print("sakshi chavhan")
#A simple print statement to display the message "sakshi chavhan"

model=YOLO('yolov8n.pt') #YOLOv8n means nano version
# loading the YOLOv8n (nano version) weights from the 'yolov8n.pt' file.


results=model('../Images/1.jpg',show=True)
#Performs object detection on the image located at '../Images/1.jpg'.
#The show=True parameter is used to display the results.

cv2.waitKey(0) #adding delay , previously we don't see output because we dont added any delay
#This is used to keep the window showing the results open.
#A value of 0 means that the program will wait indefinitely until a key is pressed.