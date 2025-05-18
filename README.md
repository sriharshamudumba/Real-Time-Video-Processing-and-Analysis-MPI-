# Real-Time Vehicle Number Plate Detection System (MPI + Serial)

This project implements both serial and MPI-parallelized versions of a vehicle number plate detection system using OpenCV and EasyOCR. It is designed for deployment on Nova cluster nodes or local machines.

---

##  Introduction

The capacity to detect license plates efficiently and accurately in video streams is pivotal for applications such as automated toll collection, traffic law enforcement, smart parking systems, and urban surveillance.

---

##  Methodology

The project includes:
- A **serial implementation** for baseline performance
- A **parallel MPI implementation** to achieve scalability and runtime reduction

---

##  Technical Stack

- **Languages**: Python
- **Libraries**: OpenCV, EasyOCR, NumPy, mpi4py
- **Environment**: Nova HPC Cluster with MPI

---

##  Workflow Overview

1. **Video Load**: Frames are extracted using `cv2.VideoCapture()`
2. **Preprocessing**: Each frame undergoes grayscale conversion, Gaussian blur, Canny edge detection, and dilation.
3. **OCR**: EasyOCR is used to detect and decode text from candidate license plate regions.
4. **Annotation**: Detected plates are highlighted with bounding boxes and annotated with text.
5. **Output**: Results are written to an annotated video and a text file.

---

##  Performance

| Video Length | Serial Time | Parallel Time (16 MPI Ranks) |
|--------------|-------------|-------------------------------|
| 10.5 sec     | 602.4 sec   | 54.3 sec                      |
| 1.5 min      | 1002.34 sec | 106.21 sec                    |
| 2.45 min     | 2435.55 sec | 132.4 sec                     |

Parallelization with MPI achieved over **18x speedup** on large video files.

---

##  Running the Code

Install dependencies:

```bash
pip install -r requirements.txt
