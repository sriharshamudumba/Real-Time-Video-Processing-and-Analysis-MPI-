import cv2
import easyocr
import os
from concurrent.futures import ThreadPoolExecutor
import os

# Set CUDA_VISIBLE_DEVICES to force GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU with index 0 (change if necessary)


# Path to the input video file
video_path = "S:/EDU/MS Computer Engineering/CPRE_MATH525 High Performance Comuting/Traffic_monitoring/Math525/Traffic.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize EasyOCR reader with GPU
reader = easyocr.Reader(['en'], gpu=True)

# Create a directory to save the processed frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Initialize the total vehicle count
total_vehicle_count = 0

# Define a function to process a single frame
def process_frame(frame):
    global total_vehicle_count
    # Detect license plates in the frame
    results = reader.readtext(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Process the OCR results
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Only consider results with high confidence
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers if needed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame as an image
    frame_filename = os.path.join(output_dir, f"frame_{total_vehicle_count}.jpg")
    cv2.imwrite(frame_filename, frame)

    # Increment the total vehicle count
    total_vehicle_count += 1

# Process frames using a ThreadPoolExecutor with 4 threads
with ThreadPoolExecutor(max_workers=4) as executor:
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Submit the frame for processing
        executor.submit(process_frame, frame)

# Release resources
cap.release()

# Save the total vehicle count to a text file
with open("number_plates.txt", "w") as f:
    f.write(f"Total Vehicles: {total_vehicle_count}\n")

print(f"Total Vehicles: {total_vehicle_count}")
print(f"Processed frames saved in '{output_dir}' directory.")
