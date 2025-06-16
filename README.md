# Real-Time Deepfake Detection Application

This application uses MesoNet and FastAPI to detect deepfakes in uploaded videos, images, or live webcam streams.

## Features

- Upload videos for deepfake analysis
- Upload images for instant deepfake detection
- Live webcam deepfake detection
- Per-frame confidence scores
- Visual timeline of detection results
- Face detection and highlighting

## Installation

### Prerequisites
- Python 3.8 or higher
- macOS users with Apple Silicon (M1/M2/M3) require special TensorFlow installation

### Setup Instructions

1. Clone this repository
   ```bash
   git clone <repository-url>
   cd deepfakerealtimedetection
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages
   ```bash
   # For Intel-based systems
   pip install -r requirements.txt
   
   # For Apple Silicon (M1/M2/M3) Macs
   # Note: You might need to install packages individually if you encounter issues
   pip install fastapi uvicorn
   pip install numpy opencv-python mediapipe python-multipart pillow jinja2 aiofiles
   pip install tensorflow-macos tensorflow-metal
   ```

## Usage

### Full Application
Start the server:
```bash
python3 run.py
```
Then open your browser and go to http://localhost:8000

### Using the Application
1. Upload a video file, image file, or use your webcam for real-time detection
2. View the analysis results showing confidence scores for deepfake probability
3. For videos, examine the frame-by-frame timeline for detailed analysis
4. For images, see highlighted faces with individual confidence scores

## Technologies Used

- FastAPI for the backend API
- MesoNet model for deepfake detection
- MediaPipe for face detection
- ONNX Runtime for optimized inference (when available)
- JavaScript/HTML for the frontend interface

## Troubleshooting

- If you encounter "Module not found" errors, ensure all dependencies are installed
- For Apple Silicon Macs, use the TensorFlow Metal version for GPU acceleration
- If port 8000 is unavailable, modify the port in run.py