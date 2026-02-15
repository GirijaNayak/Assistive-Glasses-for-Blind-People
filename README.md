# 🕶️ Assistive Glasses for Blind People

An AI-powered multimodal assistive system designed to help visually impaired individuals by combining:

- 🎯 Object Detection (YOLOv8)
- ✋ Gesture Recognition (MobileNetV2)
- 🔊 Environmental Sound Recognition (PANNs CNN14 + ESC-50)
- 🧠 Multimodal Fusion Layer

---

## 🚀 Project Overview

This project integrates computer vision and audio intelligence into a unified system that can:

- Detect real-world objects using a camera
- Recognize hand gestures for interaction
- Identify environmental sounds
- Fuse visual + audio information for contextual awareness

The goal is to simulate smart assistive glasses capable of real-time perception and feedback.

---

## 🧠 Models Used

### 🎯 Object Detection
- YOLOv8 (Ultralytics)
- Custom training supported

### ✋ Gesture Recognition
- MobileNetV2 (Transfer Learning)
- HaGRID Classification Dataset

### 🔊 Audio Recognition
- PANNs (CNN14)
- ESC-50 Environmental Sound Dataset

---

## 📂 Project Structure

├── object_detection/
├── gesture_module/
├── audio_module/
├── fusion_layer/
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Assistive-Glasses-for-Blind-People.git
cd Assistive-Glasses-for-Blind-People
pip install -r requirements.txt
