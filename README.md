# Car Surveillance System

A deep learning-based vehicle detection and classification system that leverages OpenCV for image processing and TensorFlow Keras for machine learning. This project classifies vehicles from predefined types (e.g., Camry, Minivan, SUV) and distinguishes them from random blobs (representing noise or non-vehicle objects) in video feeds.

## Project Overview

### Project Description

The **Car Surveillance System** is an intelligent vehicle detection and classification program that uses computer vision and machine learning to identify vehicles in video feeds. The system operates in two key phases: **image pre-processing** and **vehicle classification**. First, video frames are captured and pre-processed using OpenCV. A background subtraction algorithm detects moving objects (likely vehicles), isolating them from the background. Next, each detected object is analyzed using a pre-trained neural network to classify it into vehicle types such as *Camry*, *Minivan*, or *SUV*. The neural network is trained on a dataset of binary images, and after processing, the system highlights vehicles in the video feed with bounding boxes. This system can be used for surveillance, traffic monitoring, and potentially automated traffic analysis systems.

The core components of the system include:
- **Vehicle Image Dataset**: A set of vehicle images (Camry, Minivan, SUV) used to train the model.
- **Background Subtraction**: A technique to detect moving objects in video frames, helping identify vehicles.
- **Neural Network Model**: A deep learning model trained to classify vehicle types based on image features.
- **Real-Time Vehicle Detection**: The system processes video frames, detects vehicles, and classifies them in real-time, marking them with bounding boxes.

This project demonstrates how OpenCV and TensorFlow can be combined to solve real-world problems like traffic monitoring and automated vehicle classification.

---

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- numpy
- matplotlib
- pandas
- seaborn
- scikit-learn

  /Car-Surveillance/
│
├── data/
│   ├── training/
│   │   ├── vehicle_images/
│   │   │   ├── camry_binary/
│   │   │   ├── minivan_binary/
│   │   │   ├── suv_binary/
│   │   └── vehicles/                 # Dataset of various vehicle types
│   └── videos/                       # Folder containing video files (e.g., 'vid1.mp4')
├── main.py                           # Main script to train the neural network
├── threshold.py                      # Background subtraction and vehicle detection in video
├── main.ipynb                        # Jupyter notebook for training and inference
└── README.md                         # This file


## Usage
1. Training the Neural Network
The main.py script trains a neural network to classify vehicle images. The neural network learns to differentiate between images of vehicles (Camry, Minivan, SUV) and random binary blobs (representing noise).

To run the training process:
python main.py

The model will train for 30 epochs, progressively improving its accuracy in distinguishing between vehicles and random blobs. The trained model is then used for video vehicle detection.

## 2. Vehicle Detection in Video
The threshold.py script applies background subtraction to detect moving objects (likely vehicles) in the video feed. The neural network trained in main.py is then used to classify detected moving objects (cars) in real-time. The system processes each frame of the video, applies the necessary pre-processing, and highlights detected vehicles with green bounding boxes.

To run vehicle detection on a video:
python threshold.py

The system will process the video frame by frame, apply background subtraction, and detect vehicles in real-time. The output video with detected vehicles will be saved as output_video.mp4.

## 3. Jupyter Notebook Workflow
You can also use the main.ipynb notebook for training and inference, offering an interactive interface for visualizing model performance and processing videos.

Open main.ipynb using Jupyter Notebook.
Train the model and test vehicle detection directly from the notebook.

## 4. Custom Video Processing
If you want to process a different video, modify the video variable in main.py or threshold.py to point to the correct file.
video = 'path/to/your/video.mp4'


### Implementation & Results
This project successfully developed a pipeline for vehicle classification using machine learning techniques. The two-phase approach demonstrated the feasibility of combining ANN and SVM for improved accuracy. The model trained with images of different vehicle types and used background subtraction to detect vehicles in video footage. The detection results were promising, with vehicles correctly identified in most frames of the video feed.

In our tests, the model showed a high degree of accuracy in detecting and classifying cars against a variety of backgrounds. However, future work could focus on deploying the solution for real-time traffic management and improving the model’s generalization to handle a wider variety of vehicles and conditions.

## Conclusion
This project showcases the application of computer vision and machine learning to vehicle detection and classification. By leveraging OpenCV for image processing and TensorFlow Keras for neural network training, the system demonstrates real-time vehicle detection with good accuracy. It has significant potential for use in intelligent traffic systems and automated vehicle monitoring.

Contributors
Samuel Harry Axler: Project planning, data preprocessing, and video processing.
Robert Jajko: Neural network development and training, model evaluation.
Rahul Reji: Background subtraction algorithm, vehicle detection improvements.
David J. Wallington: Documentation, system integration, and testing.
