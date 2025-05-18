from mpi4py import MPI
import cv2, easyocr
import os
import numpy as np
from utils.image_processing import preprocess_frame
from utils.ocr_wrapper import detect_text

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

INPUT_VIDEO = 'input_video/sample.mp4'
OUTPUT_TEXT = 'output/output.txt'
OUTPUT_VIDEO = 'output_video/annotated_parallel_output.avi'

os.makedirs('output', exist_ok=True)
os.makedirs('output_video', exist_ok=True)

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

reader = easyocr.Reader(['en'])

# Step 1: Load and distribute frames
if rank == 0:
    all_frames = load_video_frames(INPUT_VIDEO)
    chunks = [all_frames[i::size] for i in range(size)]
else:
    chunks = None

local_frames = comm.scatter(chunks, root=0)
local_annotated = []
local_texts = []

# Step 2: Local processing
for frame in local_frames:
    processed = preprocess_frame(frame)
    boxes = detect_text(processed, reader)

    for box, text in boxes:
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)
        cv2.putText(frame, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        local_texts.append(text)
    local_annotated.append(frame)

# Step 3: Gather everything to root
gathered_frames = comm.gather(local_annotated, root=0)
gathered_texts = comm.gather(local_texts, root=0)

# Step 4: Root writes video and text
if rank == 0:
    flat_frames = [frame for sublist in gathered_frames for frame in sublist]
    flat_texts = [t for sublist in gathered_texts for t in sublist]

    if flat_frames:
        h, w = flat_frames[0].shape[:2]
        out_vid = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), 20, (w, h))
        for f in flat_frames:
            out_vid.write(f)
        out_vid.release()

    with open(OUTPUT_TEXT, 'w') as f:
        for t in flat_texts:
            f.write(f"{t}\n")
