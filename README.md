cat > README.md << 'EOF'
# 🌿 Plant Disease Prediction

A deep learning web application that detects plant diseases from leaf images using TensorFlow and Streamlit.

## 📋 About
This project uses a CNN model trained on 87,000+ images across 38 plant disease categories from the PlantVillage dataset.

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/KaungSettNaingKSN/Plant_Disease_Prediction.git
cd Plant_Disease_Prediction
```

### 2. Create conda environment
```bash
conda create -n tensorflow_env python=3.11
conda activate tensorflow_env
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 4. Download the trained model
Download `trained_model.keras` and place it in the project root folder.
[Google Drive Link] ← add your link here

### 5. Run the app
```bash
streamlit run main.py
```

## 📁 Project Structure

Plant_Disease_Prediction/
├── train/                        # Training images
├── valid/                        # Validation images
├── test/                         # Test images
├── Train_plant_disease.ipynb     # Model training notebook
├── Test_Plant_Disease.ipynb      # Model testing notebook
├── main.py                       # Streamlit web app
├── trained_model.keras           # Trained CNN model
├── training_hist.json            # Training history
└── requirement.txt               # Dependencies

## 🌱 Supported Plants & Diseases
38 classes including Apple, Blueberry, Cherry, Corn, Grape, Orange, 
Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato 
and their diseases.

## 📊 Model Architecture
- 5 Convolutional blocks (32→64→128→256→512 filters)
- Dropout layers to prevent overfitting
- Dense layer with 1500 neurons
- Softmax output for 38 classes
- Trained for 10 epochs with Adam optimizer (lr=0.0001)

## 🛠️ Tech Stack
- Python 3.11
- TensorFlow 2.17.1
- Streamlit
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn
EOF

