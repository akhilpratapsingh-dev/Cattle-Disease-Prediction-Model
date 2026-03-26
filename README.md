---
title: Cattle Disease Detector
emoji: 🐄
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: false
URL: https://akhiil1-cattle-disease-model.hf.space
---

# 🐄 Cattle Skin Disease Classification using Deep Learning


---

## 📝 Overview

This project is a **Deep Learning–based Cattle Skin Disease Classification System** that identifies multiple cattle skin diseases from images.  
It uses **MobileNetV2**, **ResNet50**, **NASNetMobile**, and a **Custom CNN**, trained on a folder-based image dataset using **TensorFlow/Keras**.

The system supports **automatic preprocessing**, **augmentation**, **class balancing**, and **model evaluation**, and saves all trained artifacts for reuse.

---

## 🚀 Features

✨ **Automatic dataset loading** from folder-structured classes  
🧪 **Train / Validation / Test split** (70% / 15% / 15%)  
🏷️ **Label encoding** for disease classes  
🖼️ **Image preprocessing + augmentation**  
🤖 **Multiple model architectures:**
- MobileNetV2 (default)
- ResNet50
- NASNetMobile
- Custom CNN
- IMAGES
- <img width="1080" height="890" alt="Screenshot 2026-03-26 202342" src="https://github.com/user-attachments/assets/10cbc1a1-9997-4204-99bc-812c26b61923" />
<img width="1858" height="969" alt="Screenshot 2026-03-25 221959" src="https://github.com/user-attachments/assets/1235aa60-c628-4192-9f90-88cab59f67c0" />



⚖️ **Handles class imbalance** using class weights  
🛠️ **Training callbacks included:**
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

📊 **Evaluation includes:**
- Accuracy
- Classification Report
- Confusion Matrix (saved as `.png`)

💾 **Saves the following outputs:**
- Trained model (`cattle_skin_disease_model.h5`)
- Label encoder (`label_encoder.pkl`)
- Confusion matrix image (`confusion_matrix.png`)

---

## 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Preprocessing** | OpenCV, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | Scikit-learn, Pickle |

---

## 📂 Project Structure

```bash


Cattle-Disease-Prediction-Model/
├── app.py                        ← Web application (Streamlit)
├── predict.py                    ← Command line prediction script
├── best_cattle_disease_model/    ← Trained model files
├── best_cattle_disease_model.h5  ← Model weights
├── dataset.csv                   ← Dataset labels
├── final_confusion_matrix.png    ← Model evaluation results
└── requirements.txt              ← Required libraries
```

---

## 🧩 How It Works

1. 📥 **Dataset Loading**  
   All images are loaded from the `combined/` directory where each subfolder represents a disease.

2. 🖼️ **Preprocessing & Augmentation**  
   Images are resized, normalized, and augmented for robustness.

3. 🤖 **Model Selection & Training**  
   The model is trained using MobileNetV2.  

4. 🧪 **Evaluation**  
   The model outputs accuracy, a classification report, and a confusion matrix.

5. 💾 **Saving Model Artifacts**  
   The trained model, label encoder, and confusion matrix plot are stored for later inference.

---
## 🌐 Live Demo (Streamlit App)

You can try the deployed version of this model through an interactive **Streamlit web application**.
*Live App:*  
👉 https://akhiil1-cattle-disease-model.hf.space
### 🔍 What the App Does
- Upload a cattle skin image 📸  
- Automatically preprocess the image  
- Predict the *skin disease class* using the trained deep learning model  
- Display the predicted label in real time  

This deployment demonstrates how the trained model can be used for *real-world inference* through a simple and user-friendly interface.

## 🎯 Future Enhancements

🚀 Add lesion detection using **YOLO/Detectron2**  
📱 Mobile app integration for field cattle diagnosis  
📈 Deploy as a **web app** using Streamlit or FastAPI  
🧬 Add **Vision Transformers (ViT)** for higher accuracy  
🌐 Expand dataset with real-world variations  

---

## ❤️ Acknowledgements

Special thanks to dataset providers supporting livestock disease identification.
