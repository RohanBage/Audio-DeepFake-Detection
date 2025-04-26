# Audio DeepFake Detection



## 🔍 Project Overview

**Audio DeepFake Detection** is an intelligent system designed to classify audio files as **real** or **fake**.  
It uses **machine learning** techniques to analyze audio feature sets and predict authenticity, providing a simple and interactive **Streamlit**-based user interface for real-time testing.

This project primarily leverages:
- A **Random Forest Classifier** trained on extracted audio features (e.g., MFCCs, Spectral Centroid, Zero-Crossing Rate, etc.).
- A **CSV-based dataset** of real and fake audio samples.
- **Streamlit** for a lightweight, web-based UI to upload `.wav` files and view predictions instantly.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Scikit-learn** (for Random Forest Model)
- **Streamlit** (for web application UI)
- **Librosa** (for audio feature extraction)
- **Pandas** & **NumPy** (for data handling)

---

## 🚀 Features

- Upload an audio `.wav` file and get instant prediction (Real / Fake).
- Audio feature extraction on the fly.
- Fast inference using a pre-trained Random Forest model.
- Clean and user-friendly Streamlit interface.

---

## 📂 Project Structure

```
DEEPFAKE DETECTION/
│
├── .venv/                          # Virtual environment (ignored by Git)
│
├── AudioSamples/                   # Test audio samples
│   ├── FAKE/                        # Folder containing fake audio samples (.wav)
│   └── REAL/                        # Folder containing real audio samples (.wav)
│
├── .gitignore                      # Git ignore file
├── app_1.py                        # Main Streamlit app (UI for uploading and prediction)
├── newTrain_1.py                   # Script for training RandomForest model
│
├── audio_deepfake_model.pkl        # Trained RandomForest model (saved after training)
├── feature_scaler.pkl              # Scaler used for feature normalization during training
├── scaler.pkl                      # (Seems duplicate - can merge with feature_scaler.pkl if same)
│
├── confusion_matrix.png            # Visualization of model performance (confusion matrix)
├── DATASET-balanced.csv            # Feature dataset (extracted features + labels)
│
├── requirements.txt                # List of required Python packages
└── README.md                       # Project documentation (you'll add this)

```

---

## 📈 How It Works

1. **Feature Extraction**:  
   When an audio file is uploaded, features like MFCCs, Chroma, Spectral Centroid, etc., are extracted using `librosa`.
   
2. **Model Prediction**:  
   The extracted feature vector is passed to a **Random Forest Classifier** trained previously on a labeled dataset.

3. **Result Display**:  
   The app immediately displays whether the audio is **REAL** or **FAKE** based on the model's prediction.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RohanBage/Audio-DeepFake-Detection.git
cd Audio-DeepFake-Detection
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Example

> Upload a `.wav` file in the app.  
> You will get an output like:

✅ **Prediction: Real Audio**

or

❌ **Prediction: Fake Audio**

---

## 🤔 Future Work

- Improve model accuracy using deep learning models (e.g., CNNs or RNNs).
- Handle additional audio formats (e.g., MP3, FLAC).
- Provide probability/confidence scores with predictions.
- Train with larger and more diverse datasets (e.g., SceneFake, FoR datasets).

---


## 📬 Contact

Made with ❤️ by [Rohan Bage](https://github.com/RohanBage)

