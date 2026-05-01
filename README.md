# AegisTrack — Real-Time Target Tracking, Prediction and AI-Based Detection System

AegisTrack is a real-time computer vision system that combines AI object detection, multi-object tracking, motion prediction, and threat analysis into a single pipeline.

This project simulates a simplified defense-style radar intelligence system using computer vision and deep learning techniques.

## Features

AI Object Detection (YOLOv8)
Detects real-world objects such as people and vehicles in real-time using a pretrained deep learning model.

Multi-Object Tracking
Assigns unique IDs to detected objects and maintains identity across frames.

Trajectory Tracking
Stores movement history of each object and visualizes paths using motion trails.

Motion Prediction
Predicts future position based on recent movement using velocity estimation.

Threat Analysis System
Classifies targets based on speed into LOW, MEDIUM, and HIGH threat levels.

Logging and Analytics
Exports real-time data to a CSV file including object ID, class label, position, and speed.

## System Pipeline

Video Input → YOLO Detection → Tracking → History → Prediction → Threat Analysis → Logging

## Tech Stack

Python
OpenCV
NumPy
Ultralytics YOLOv8
CSV for logging

## Installation

pip install opencv-python numpy ultralytics

## Usage

python main.py

Press ESC to exit.

## Output

Live Video Feed
Displays bounding boxes with object class, unique ID, and threat level. Shows motion trajectory and predicted position.

CSV Log File
A file named tracking_log.csv is generated with the following format:

Frame,ID,Class,X,Y,Speed
12,0,person,320,210,4.5
13,0,person,330,215,5.1

## Limitations

Tracking is distance-based and may fail with fast or overlapping objects.
Prediction assumes linear motion and is not accurate for erratic movement.
Performance depends on hardware capabilities.

## Future Improvements

Integrate DeepSORT for more robust tracking
Use Kalman Filter for better prediction
Add radar-style UI visualization
Implement object re-identification
Build analytics dashboard using Streamlit

## Learning Outcomes

This project demonstrates real-time computer vision pipelines, integration of deep learning with classical methods, time-series motion analysis, and system-level design.

## Author

Haziq Mubashir
BS Data Science Student

## Feedback

Feedback and suggestions are welcome.
