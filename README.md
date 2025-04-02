# DontFall

DontFall is an Edge Computing and Analytics project. The project aims to run real-time Fall Detection using stimulus from a camera, an MPU 6050 accelerometer and a microphone. The project is designed to run on a Raspberry Pi 400.

![Architecture Diagram](assets/image.png)

## Peer-to-Peer (P2P) Communication

In this setup, all edge devices (2× Raspberry Pi 400 and 1× M5Stick C Plus) communicate over a shared mobile hotspot using Python sockets instead of MQTT. The choice of sockets allows for direct peer-to-peer TCP communication without needing a broker, which is ideal in a local intranet environment.

**Why Python Sockets?**
* Low Latency: Direct TCP connections reduce message overhead and response time.

* No Broker Needed: Simplifies architecture and eliminates single points of failure.

* Custom Protocols: Allows full control over data format, flow, and error handling—great for real-time analytics.

* Lightweight: Minimal resource usage, perfect for constrained edge devices.

In contrast, MQTT—though excellent for cloud-connected IoT—is less efficient in this closed, local setup due to its reliance on a central broker and added protocol complexity.


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

