# Currency Recognition Model > Version -3 YOLOv11 model
https://universe.roboflow.com/daniel-raj-c/currency-recognition-model-g16wl

Provided by a Roboflow user
License: MIT

The Currency Recognition Model is a computer vision system built using instance segmentation to detect and identify various denominations of Indian currency notes in images. The model can recognize and segment multiple notes in a single frame, even in cluttered environments or under varying lighting conditions.

## 💰 Recognized Currency Classes:

- ₹0 (sample/mock or defaced notes)
- ₹1
- ₹2
- ₹5
- ₹10
- ₹20
- ₹50
- ₹100
- ₹200
- ₹500

Each class represents a specific Indian currency denomination. The model assigns a class label and draws a segmentation mask around each detected note.

## 📦 Dataset Details:
- Collected under varied lighting, backgrounds, and note orientations
- Annotated with precise segmentation masks and class labels
- Supports overlapping notes in a single frame

## 🧾 Key Features:
- Multi-class detection in real-world scenarios
- Accurate instance-wise segmentation
- Suitable for low-light and complex backgrounds
- Robust against note folds, wrinkles, and partial occlusions

## 🚀 Applications:
- Smart currency counters and kiosks
- Assistive tools for visually impaired individuals
- Currency classification in mobile apps
- Educational tools for recognizing money
- Banking and ATM automation