# 🌿 Plant Disease Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A deep learning web application that detects **38 plant diseases** from leaf images in seconds — powered by a custom CNN trained on 87,000+ images.

---

## ✨ Demo

Upload any plant leaf image and instantly get:
- 🔍 **Disease name** detected by the model
- 🌱 Works with **14 different crops**
- ⚡ Results in **under 2 seconds**

---

## 🚀 Getting Started

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Git

### 1. Clone the repository
```bash
git clone https://github.com/KaungSettNaingKSN/Plant_Disease_Prediction.git
cd Plant_Disease_Prediction
```

### 2. Create & activate conda environment
```bash
conda create -n tensorflow_env python=3.11
conda activate tensorflow_env
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 5. Launch the app
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Plant_Disease_Prediction/
│
├── 📂 train/                       # Training images (70,295)
├── 📂 valid/                       # Validation images (17,572)
├── 📂 test/                        # Test images (33)
│
├── 📓 Train_plant_disease.ipynb    # Model training notebook
├── 📓 Test_Plant_Disease.ipynb     # Model testing & evaluation
│
├── 🐍 main.py                      # Streamlit web app
├── 🧠 trained_model.keras          # Trained CNN model weights
├── 📊 training_hist.json           # Training accuracy/loss history
├── 🖼️  home_page.jpeg              # App homepage image
└── 📋 requirement.txt              # Python dependencies
```

---

## 🧠 Model Architecture

```
Input (128×128×3)
    │
    ├── Conv2D(32) → Conv2D(32) → MaxPool
    ├── Conv2D(64) → Conv2D(64) → MaxPool
    ├── Conv2D(128) → Conv2D(128) → MaxPool
    ├── Conv2D(256) → Conv2D(256) → MaxPool
    ├── Conv2D(512) → Conv2D(512) → MaxPool
    │
    ├── Dropout(0.25)
    ├── Flatten
    ├── Dense(1500, relu)
    ├── Dropout(0.4)
    │
    └── Dense(38, softmax) → Prediction
```

**Training config:** Adam optimizer · lr=0.0001 · 10 epochs · categorical crossentropy

---

## 🌱 Supported Plants & Diseases

| Plant | Conditions |
|-------|-----------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍊 Orange | Haunglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫐 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow 2.17.1 | Model training & inference |
| Keras | Neural network layers |
| Streamlit | Web application UI |
| OpenCV | Image reading & processing |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| scikit-learn | Evaluation metrics |

---

## 📊 Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Total images:** ~87,000 RGB leaf images
- **Classes:** 38 (healthy + diseased categories)
- **Split:** 80% train / 20% validation

---

## 👤 Author

**Kaung Sett Naing**
- GitHub: [@KaungSettNaingKSN](https://github.com/KaungSettNaingKSN)

---

*Built with ❤️ for smarter agriculture*
