# 🎛️ Hand Gesture Volume Controller

Control your system's volume using hand gestures with the help of OpenCV, MediaPipe, and Pycaw!

This Python project captures real-time video input, detects your hand landmarks using MediaPipe, and adjusts your system volume based on the distance between your thumb and index finger.

---

## 🚀 Features

- Real-time hand tracking using MediaPipe
- Adjust system volume using hand gestures
- Visual feedback with hand landmarks and a volume bar
- Works offline with a standard webcam

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- Pycaw (Windows only)
- NumPy

---

## 🔧 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adnaan-786/Volume-Control.git
   cd volume-control

	2.	Install dependencies

pip install opencv-python mediapipe comtypes pycaw numpy


	3.	Run the script

python volume.py


	4.	Usage
	•	Show your hand in front of your webcam
	•	Move your thumb and index finger closer or farther apart
	•	The system volume will increase or decrease based on the distance
	•	Press q to exit

⸻

🖥️ Compatibility
	•	✅ Windows (required for Pycaw audio control)
	•	❌ Not fully supported on macOS or Linux

⸻

📂 Project Structure

.
├── volume.py           # Main Python script
├── README.md           # Project documentation


⸻

📸 Demo

Add a GIF or screenshot of your project in action here

⸻

🙋‍♂️ Author

Created by Adnaan Gouri

⸻

📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
