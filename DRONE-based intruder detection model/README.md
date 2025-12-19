# Drone-Based Intruder Detection Model

## 📌 Project Overview
This project implements a drone-based intruder detection system using computer vision techniques. The model is developed to process aerial imagery and detect potential intruders by leveraging object detection workflows. The project focuses on dataset preparation, annotation conversion, and image processing as part of a machine learning pipeline.

The implementation uses the VisDrone dataset and prepares data in a YOLO-compatible format for object detection experiments.

---

## 🎯 Objectives
- Understand drone-based surveillance and intrusion detection concepts  
- Work with large-scale aerial image datasets  
- Convert annotation formats for object detection models  
- Apply computer vision techniques for real-world security use cases  

---

## 🗂 Dataset
- **Dataset Used:** VisDrone Dataset  
- The dataset is downloaded programmatically and organized locally  
- Annotation files are processed and converted into YOLO format  

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Dataset Handling:** KaggleHub  
- **Annotation Format:** YOLO  
- **Libraries:** NumPy, OS, Shutil  

---

## ⚙️ System Workflow
1. Download VisDrone dataset using KaggleHub  
2. Organize images and annotations into structured directories  
3. Convert VisDrone annotation format into YOLO-compatible labels  
4. Process aerial images for object detection readiness  
5. Prepare data for training and experimentation with detection models  

---

## 📂 Project Structure
DRONE-based intruder detection model/
│
├── images/ # Input drone images
├── labels/ # YOLO-formatted annotation files
├── processed_images/ # Processed / resized images
├── detection_model.ipynb
└── README.md
