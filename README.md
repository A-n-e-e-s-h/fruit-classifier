# 🍓 Fruit Classification Web App with Calorie Estimator

This is a Flask-based web application that uses a deep learning model to classify fruits from images and estimate their calorie content based on weight (grams).

---

## 🚀 Features

- 📷 Upload an image of a fruit
- 🧠 Classifies into:
  - Apple, Banana, Grape, Mango, Strawberry, Chickoo, Kiwi, Orange
- 🧮 Estimates calories based on weight input
- 💡 Built with transfer learning (MobileNetV2)

---

## 🧠 Machine Learning Model

- ✅ **Base Model**: MobileNetV2 (pretrained on ImageNet)
- ✅ **Transfer Learning**: Fine-tuned on 8 fruit classes
- ✅ **Input size**: 224x224 RGB images
- ✅ **Framework**: TensorFlow / Keras

---

## 🛠️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fruit-classifier.git
cd fruit-classifier


## 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux



## 3. Install dependencies
pip install -r requirements.txt




## 4. Run the app
cd app
python main.py




