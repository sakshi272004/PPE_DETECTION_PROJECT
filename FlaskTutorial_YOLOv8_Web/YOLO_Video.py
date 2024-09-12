import mysql.connector
from ultralytics import YOLO
import cv2
import math
from datetime import datetime



# Function to establish database connection
def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="pythonmysql"
        )
        print("Database connection successful")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def save_detection_data(image_id, detected_hardhats, detected_masks, detected_persons, detected_safety_vests,
                        detected_machinery, detected_vehicles,
                        hardhat_confidence, mask_confidence, safety_vest_confidence,
                        machinery_confidence, vehicle_confidence):
    conn = connect_to_database()
    if conn:
        try:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Calculate adjusted confidence scores based on presence or absence
            hardhat_confidence = 1 - hardhat_confidence if detected_hardhats > 0 else 0 - hardhat_confidence
            mask_confidence = 1 - mask_confidence if detected_masks > 0 else 0 - mask_confidence
            safety_vest_confidence = 1 - safety_vest_confidence if detected_safety_vests > 0 else 0 - safety_vest_confidence
            machinery_confidence = 1 - machinery_confidence if detected_machinery > 0 else 0 - machinery_confidence
            vehicle_confidence = 1 - vehicle_confidence if detected_vehicles > 0 else 0 - vehicle_confidence

            # Insert data into ppe_detection table
            sql = """
            INSERT INTO ppe_detection 
            (image_id, timestamp, detected_hardhats, detected_masks, detected_persons, detected_safety_vests, 
             detected_machinery, detected_vehicles, 
             hardhat_confidence, mask_confidence, safety_vest_confidence, machinery_confidence, vehicle_confidence) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (image_id, timestamp, detected_hardhats, detected_masks, detected_persons, detected_safety_vests,
                                 detected_machinery, detected_vehicles,
                                 hardhat_confidence, mask_confidence, safety_vest_confidence,
                                 machinery_confidence, vehicle_confidence))
            conn.commit()

            print("Detection data saved to database successfully")

        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            cursor.close()
            conn.close()


# Function to perform video detection
def video_detection(path_x):
    cap = cv2.VideoCapture(path_x);
    frame_width = int(cap.get(3));
    frame_height = int(cap.get(4));

    model = YOLO("../Project-PPE detection/ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']

    # Dictionary to track presence of PPE kits and their confidence scores
    ppe_presence = {
        'Hardhat': (False, 0.0),
        'Mask': (False, 0.0),
        'Safety Vest': (False, 0.0),
        'Safety Cone': (False, 0.0),
        'machinery': (False, 0.0),
        'vehicle': (False, 0.0)
    }

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        # Reset presence flags and confidence scores before processing each frame
        for ppe in ppe_presence:
            ppe_presence[ppe] = (False, 0.0)

        # Variables to store detection counts and confidence scores
        detected_hardhats = 0
        detected_masks = 0
        detected_safety_vests = 0
        detected_persons = 0
        detected_machinery = 0
        detected_vehicles = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Update presence and count of detected items
                if class_name == 'Hardhat' and conf > 0.5:
                    ppe_presence['Hardhat'] = (True, conf)
                    detected_hardhats += 1
                elif class_name == 'Mask' and conf > 0.5:
                    ppe_presence['Mask'] = (True, conf)
                    detected_masks += 1
                elif class_name == 'Safety Vest' and conf > 0.5:
                    ppe_presence['Safety Vest'] = (True, conf)
                    detected_safety_vests += 1
                elif class_name == 'Person' and conf > 0.5:
                    detected_persons += 1
                elif class_name == 'machinery' and conf > 0.5:
                    ppe_presence['machinery'] = (True, conf)
                    detected_machinery += 1
                elif class_name == 'vehicle' and conf > 0.5:
                    ppe_presence['vehicle'] = (True, conf)
                    detected_vehicles += 1

                # Draw bounding boxes and labels for detections with confidence > 0.5
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = (0, 204, 255) if class_name == 'Hardhat' else (222, 82, 175) if class_name == 'Mask' else (
                    0, 149, 255) if class_name == 'Safety Vest' else (85, 45, 255)

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Construct image_info string for database storage
        image_id = f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}"

        # Extract confidence scores for each item
        hardhat_confidence = ppe_presence['Hardhat'][1]
        mask_confidence = ppe_presence['Mask'][1]
        safety_vest_confidence = ppe_presence['Safety Vest'][1]
        machinery_confidence = ppe_presence['machinery'][1]
        vehicle_confidence = ppe_presence['vehicle'][1]

        # Print presence or absence of each PPE after processing detections
        for ppe, (present, confidence) in ppe_presence.items():
            if present:
                print(f"{ppe} is present with confidence {confidence}")
            else:
                print(f"{ppe} is absent")

        # Save detection data to database
        save_detection_data(image_id, detected_hardhats, detected_masks, detected_persons, detected_safety_vests,
                            detected_machinery, detected_vehicles, hardhat_confidence, mask_confidence,
                            safety_vest_confidence,
                            machinery_confidence, vehicle_confidence)

        yield img

    cap.release()
    cv2.destroyAllWindows()


# Example usage: Call video_detection function with the path to your video file
video_detection("path_to_video_file.mp4")

