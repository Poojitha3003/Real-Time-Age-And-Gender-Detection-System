# 👶🧔 Real-Time Age and Gender Detection

This project uses **OpenCV** and **pre-trained deep learning models** to detect a person's **age** and **gender** from a webcam feed in real time.

---

## 📌 Features

- Real-time face detection using Haar Cascades
- Age group prediction (e.g., 20-25, 40-65)
- Gender prediction (Male / Female)
- Uses OpenCV DNN module with Caffe pre-trained models
- Live webcam integration

---

## 🔧 Technologies Used

- Python
- OpenCV (`cv2`)
- NumPy
- Pre-trained Caffe models (`.caffemodel`, `.prototxt`)

---

## 🧠 Age & Gender Model Info

- **Age Model:** Predicts one of 9 age ranges
- **Gender Model:** Predicts either Male or Female
- Models:  
  - `age_net.caffemodel`  
  - `age_deploy.prototxt`  
  - `gender_net.caffemodel`  
  - `gender_deploy.prototxt`

---

## 📁 Project Structure

AgeGenderDetection/
│
├── models/
│ ├── age_net.caffemodel
│ ├── age_deploy.prototxt
│ ├── gender_net.caffemodel
│ └── gender_deploy.prototxt
│
├── age_gender_detection.py
└── README.md


---

## 🚀 How to Run the Project

1.**Clone the repository**

   git clone https://github.com/yourusername/age-gender-detection.git
   cd age-gender-detection
   
2.**Install dependencies**

pip install opencv-python numpy

3.**Run the project**

python age_gender_detection.py

4.**Press**

Press q to exit the webcam feed

## 📸 Screenshots

### 🎥 Real-Time Detection Output:
![Screenshot 1](Screenshot%202024-09-30%20134224.png)
![Screenshot 2](Screenshot%202024-09-30%20134454.png)


