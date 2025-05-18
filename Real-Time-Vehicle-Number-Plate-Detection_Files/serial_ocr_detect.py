import cv2
import easyocr
import os
from utils.image_processing import preprocess_frame
from utils.ocr_wrapper import detect_text

INPUT_PATH = 'input_video/sample.mp4'     # For the code to run follow the folder strcture and upload the input accordingly
OUTPUT_TEXT = 'output/output.txt'
OUTPUT_VIDEO = 'output_video/annotated_output.avi'

os.makedirs('output', exist_ok=True)
os.makedirs('output_video', exist_ok=True)

cap = cv2.VideoCapture(INPUT_PATH)
width, height = int(cap.get(3)), int(cap.get(4))
out_vid = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
reader = easyocr.Reader(['en'])

vehicle_count = 0
with open(OUTPUT_TEXT, 'w') as log:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame)
        boxes = detect_text(processed, reader)
        vehicle_count += len(boxes)

        for box, text in boxes:
            cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)
            cv2.putText(frame, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            log.write(f"{text}\n")

        out_vid.write(frame)

cap.release()
out_vid.release()
print(f"Total detected plates: {vehicle_count}")
