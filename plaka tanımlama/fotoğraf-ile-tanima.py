import cv2
import cvzone
import time

from ultralytics import YOLO

classNames = ["Plaka"]
confidence_threshold = 0.7
box_scale = 1.2

# Load an image from file
img = cv2.imread('deneme3.jpg')  # Replace 'path_to_your_image.jpg' with the actual path to your image file

model = YOLO("best.pt")

start_time = time.time()
save_img = False
conf_above_threshold_time = None

while True:
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf > confidence_threshold:
                if conf_above_threshold_time is None:
                    conf_above_threshold_time = time.time()

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                img = cv2.GaussianBlur(img, (3, 3), 0)

                width = int((x2 - x1) * box_scale)
                height = int((y2 - y1) * box_scale)
                x1 = max(0, int(x1 + (x2 - x1) * (1 - box_scale) / 2))
                y1 = max(0, int(y1 + (y2 - y1) * (1 - box_scale) / 2))
                x2 = min(img.shape[1], x1 + width)
                y2 = min(img.shape[0], y1 + height)

                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Kutu içeriğini kutu ile vurgula
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if conf > 0.75 and not save_img:
                    elapsed_time_above_threshold = time.time() - conf_above_threshold_time
                    if elapsed_time_above_threshold >= 3:
                        filename = "deneme.png"
                        # Calculate the center of the ROI
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        # Calculate the new width and height to control zoom
                        new_width = int((x2 - x1) * box_scale)
                        new_height = int((y2 - y1) * box_scale)
                        # Adjust the new x and y positions to maintain the center
                        new_x1 = max(0, center_x - new_width // 2)
                        new_y1 = max(0, center_y - new_height // 2)
                        new_x2 = min(img.shape[1], new_x1 + new_width)
                        new_y2 = min(img.shape[0], new_y1 + new_height)
                        # Resize the image to 1920 x 1080 before saving
                        new_img = cv2.resize(img[new_y1:new_y2, new_x1:new_x2], (256, 256))
                        cv2.imwrite(filename, new_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        save_img = True
                        conf_above_threshold_time = None
                else:
                    conf_above_threshold_time = None

    end_time = time.time()
    start_time = end_time

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q') or save_img:
        break

cv2.destroyAllWindows()


import easyocr

reader = easyocr.Reader(['tr'])
result = reader.readtext('deneme.png')
print(result[0][1])

