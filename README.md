# Video-Segmentation-Application (YOLOv8 + Streamlit)

This project is a web-based application that allows users to **segment objects in videos** using YOLOv8 and apply various **background effects**, such as blur, cartoon, solid color, or even custom images.

Developed with **Streamlit**, **OpenCV**, and **Ultralytics YOLOv8**, the app provides an intuitive interface for real-time video processing and result visualization.

---

## Repository Structure

  ```bash
  your-repo/
  ├── video_segmentation.py                  # Main Streamlit application script
  ├── README.md               # Project documentation (this file)
  ├── video_presentation.md  # Final 7-minute video presentation
  ├── demo_screenshots/       # Folder with UI and result screenshots
  │   ├── upload.png
  │   ├── processing.png
  │   └── result.png
```
---

## Features

-  Upload and process video files (`.mp4`, `.avi`, `.mov`)
-  Object detection and segmentation using **YOLOv8-seg**
-  Background replacement with:
      * Solid color
      * Blur effect
      * Cartoon effect
      * Transparent background
      * Custom image
-  Video preview and download after processing

---

## Model Used

The segmentation is powered by the [YOLOv8n-seg](https://docs.ultralytics.com/models/yolov8/#object-segmentation) model, a lightweight yet powerful model trained on the COCO dataset.

---

## How to Run the App Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Salmamhd/Video-Segmentation-Application.git
   cd your-repo-name
2. **Create a virtual environment and activate it**
   ```bash
   python -m venv venv
   # For Linux/MacOS:
   source venv/bin/activate
   # For Windows:
   venv\Scripts\activate
4. **Install the dependencies**
   ```bash
   pip install streamlit opencv-python-headless numpy pillow ultralytics
6. **Download the YOLOv8n-seg model (if not already downloaded)**
   ```python
   from ultralytics import YOLO
   YOLO("yolov8n-seg.pt")
8. **Run the Streamlit app**
   ```bash
   python -m streamlit run app.py

---

## Tech Stack

| Tool / Library        | Description                                           |
|------------------------|-------------------------------------------------------|
| **Python 3.8+**        | Core programming language used for development        |
| **[Streamlit](https://streamlit.io/)**        | Framework for building interactive web apps and UI        |
| **[OpenCV](https://opencv.org/)**             | For video processing, frame-by-frame object manipulation  |
| **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** | Deep learning model used for video object segmentation     |
| **[NumPy](https://numpy.org/)**               | Efficient manipulation of image arrays                    |
| **[Pillow (PIL)](https://python-pillow.org/)**| Image processing, resizing, and file format handling      |

---

## License

This project is licensed under the MIT License — free to use and adapt.
