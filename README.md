#  Laughter Detection in Video

This project provides a simple web app that allows users to upload a video file and detects segments containing **laughter** using audio analysis with the **YAMNet** deep learning model.

##  Features

- Upload `.mp4`, `.mov`, `.avi`, or `.mkv` video files.
- Extracts audio from the video.
- Uses **TensorFlow's YAMNet** model to identify laughter in audio.
- Displays laughter segments as **timestamps**.
- Visualizes the laughter timeline on a horizontal graph.

---

## Project Structure
```
laughter_detection_in_a_video/
├── app.py # Streamlit app script
├── laughter_detector.py # Laughter detection logic using YAMNet
├── requirements.txt # Required Python packages
└── README.md # Project documentation
```
#  Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/vishal-9689/laughter_detection_in_a_video.git
cd laughter_detection_app
```
## 2. Create Virtual Environment (optional but recommended)
 ```
 python -m venv venv
```
 ### On Windows
 ```
 venv\Scripts\activate
```
 ### On macOS/Linux
 ```
 source venv/bin/activate
```
## 3. Install Dependencies
```
 pip install -r requirements.txt
 ```
# Running the App
 ```
streamlit run app.py
```
 # How It Works
- moviepy extracts audio from video.
- librosa loads audio at 16kHz.
- YAMNet model (from TensorFlow Hub) detects over 500 sound classes.
- The script checks for "Laughter" class confidence and records its timestamps.
# Requirements
- Python 3.83.11
- Internet access (for downloading the YAMNet model and class map)
 Notes
- Currently supports mono-channel WAV audio.
- Detection is based on a simple threshold (adjustable in laughter_detector.py).
- Merges adjacent segments into continuous ones for cleaner results.
# License
This project is provided for educational purposes and is free to use.
