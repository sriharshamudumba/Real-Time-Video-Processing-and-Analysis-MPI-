# Real-Time-Vehicle-Number-Plate-Detection


This project uses OpenCV, EasyOCR, and MPI for detecting number plates from videos in both serial and parallel modes.

## Features
- Real-time OCR
- MPI-based frame parallelization
- OpenCV preprocessing

 ## Folder structure
├── input_video/              
├── output_video/             
├── output/                   
├── results/                  
├── utils/                    
├── serial_ocr_detect.py
├── parallel_ocr_detect_mpi.py
├── requirements.txt
└── README.md


## How to Run

```bash
pip install -r requirements.txt
mpirun -n 16 python parallel_ocr_detect_mpi.py
python serial_ocr_detect.py


