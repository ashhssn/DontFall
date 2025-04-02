# DontFall

DontFall is an Edge Computing and Analytics project. The project aims to run real-time Fall Detection using stimulus from a camera, an MPU 6050 accelerometer and a microphone. The project is designed to run on a Raspberry Pi 400.

![Architecture Diagram](assets/image.png)

### Peer-to-Peer (P2P) Communication

In this setup, all edge devices (2× Raspberry Pi 400 and 1× M5Stick C Plus) communicate over a shared mobile hotspot using Python sockets instead of MQTT. The choice of sockets allows for direct peer-to-peer TCP communication without needing a broker, which is ideal in a local intranet environment.

**Why Python Sockets?**
* Low Latency: Direct TCP connections reduce message overhead and response time.

* No Broker Needed: Simplifies architecture and eliminates single points of failure.

* Custom Protocols: Allows full control over data format, flow, and error handling—great for real-time analytics.

* Lightweight: Minimal resource usage, perfect for constrained edge devices.

In contrast, MQTT—though excellent for cloud-connected IoT—is less efficient in this closed, local setup due to its reliance on a central broker and added protocol complexity.

### Audio Detection

`AudioAnalysis.py`
- Uses a loaded fall detection model from Train.py using audio as an input from a microphone
- Captures audio and extract its feature using librosa MFCC

`Train.py`
- To train the fall detection model (Simple CNN) from labeled audio datasets (`Edge_AudioAnalysis.ipynb`) 

`Edge_AudioAnalysis.ipynb`
- Contains dataset (SAFE: Sound Analysis for Fall Event Detection) from kaggle
- Contains a csv file to automate label the dataset into 0 or 1 (Nonfall / Fall)

### Pose Detection using Camera

A MobileNetV2 model was fine-tuned on a large dataset sourced from Kaggle. The trained `.h5` model was converted to a `.tflite` format for efficient inference on a Raspberry Pi 400 using a webcam.

- TensorFlow Lite conversion reduced computational load for real-time detection.

- The Raspberry Pi 400 utilized a webcam for pose analysis.

- A microphone and accelerometer were integrated as activation triggers.

By combining multiple sensors, the system enhances accuracy while minimizing false positives; making it suitable for real-world elderly care applications.

### Accelerometer

**Data Collection and Preparation**

Sourced accelerometer data from the SisFall dataset, containing both falls and activities of daily living (ADLs)
Extracted and processed MMA8451Q sensor data (columns 6-8) from multiple subjects
Converted raw values to g units and cleaned missing/invalid data points

**Feature Engineering**

Implemented sliding window approach with 50% overlap to analyze acceleration patterns
Calculated seven key features: average magnitude, standard deviation, max/min magnitude, maximum jerk, free-fall duration, and impact peak
Normalized features using StandardScaler to improve model stability

**Model Development**

Designed a lightweight neural network (7→4→1) optimized for edge deployment
Applied L2 regularization and adaptive learning rate reduction to prevent overfitting
Trained on Google Colab using 80/20 train-test split with binary cross-entropy loss
Monitored convergence through validation accuracy and loss curves

**Edge Deployment**

Extracted optimized weights from trained model
Hard-coded weights into M5StickC Plus custom firmware for on-device inference
Deployed the code to the device using UIFlow2.0


## Installation

Clone the repository and navigate to the project directory. Install requirements using pip:

```bash
pip install -r requirements.txt
```

## Usage

All edge devices should be connected to one access point (i.e your mobile hotspot or router)

Run the accelerometer code by uploading the code found in `accelerometer` folder into the M5Stick C Plus.

Run the camera and microphone code by running the `main.py` file in the `vision` folder into one Raspberry Pi 400 connected to the camera.

Run the dashboard code by running the `main.py` file in the `root` folder into another Raspberry Pi 400.

