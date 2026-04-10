# ☀️ Solar Panel Defect Detection
live link  -- - > https://solarpaneldefectdetection.streamlit.app/
An AI-powered web application that detects defects in solar panels using deep learning and computer vision.

---

## 🚀 Overview

This project uses a **pretrained EfficientNet model (PyTorch)** to classify solar panel images into different defect categories such as dust, bird drops, physical damage, and more.

The model is deployed using **Streamlit**, providing an interactive and user-friendly interface.

---

## 🧠 Features

* 🔍 Detect multiple types of solar panel defects
* ⚡ Fast and accurate predictions using EfficientNet
* 📊 Displays prediction confidence
* 🖼️ Image upload interface
* 🎯 Top-3 predictions with probabilities
* 🎨 Modern UI built with Streamlit

---

## 🗂️ Classes

The model classifies images into the following categories:

* Bird-drop
* Clean
* Dusty
* Electrical-damage
* Physical-damage
* Snow-Covered

link https://drive.google.com/file/d/1RLzDnfDfKbqHmZdAdp3_dJ8CDEatSYjF/view?usp=drive_link
---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **Streamlit**
* **NumPy**
* **PIL (Python Imaging Library)**

---

## 📦 Model Details

* Architecture: EfficientNet-B0
* Custom Classifier Head:

  * BatchNorm
  * Linear Layer
  * ReLU
  * Dropout
  * Output Layer
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shivanshvyas1729/solar-panel-defect-detection.git
cd solar-panel-defect-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Usage

1. Upload a solar panel image
2. Model analyzes the image
3. View:

   * Predicted defect class
   * Confidence score
   * Top 3 predictions

---

## 📁 Project Structure

```
├── app.py
├── best_model.pth
├── requirements.txt
└── README.md
```

---

## 📊 Example Output

* Prediction: Dusty
* Confidence: 92%

---

## 🔮 Future Improvements

* 📸 Grad-CAM visualization (highlight defect areas)
* 🌐 Deployment on cloud (Streamlit Cloud / AWS)
* 📱 Mobile-friendly UI
* ⚡ Faster inference optimization

---

## 👨‍💻 Author

**Shivansh Vyas**

* 📧 Email: [shivanshvyas1729@example.com](mailto:shivanshvyas1729@example.com)
* 💻 GitHub: https://github.com/Shivanshvyas1729
* 💼 LinkedIn: https://www.linkedin.com/in/shivanshvyas
* 🌐 Portfolio: https://shivanshvyas1729portfolio.netlify.app/

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repository!

---
