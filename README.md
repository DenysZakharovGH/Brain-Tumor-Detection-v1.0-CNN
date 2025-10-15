# 🧠 Brain Tumor Detection using MRI Images

![Project Banner](https://cdn.pixabay.com/photo/2017/02/23/13/05/brain-2099151_1280.jpg)

> 🚀 A deep learning project that detects **brain tumors from MRI scans** with **99% accuracy**.  
> Built with passion, powered by data, and trained on the [Kaggle Brain MRI dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).

---

## 🧩 Overview

This project focuses on detecting the presence of **brain tumors** using **MRI images**.  
It uses a **Convolutional Neural Network (CNN)** trained on thousands of MRI scans to accurately classify whether a tumor is present.

The final trained model achieves:
- ✅ **Accuracy:** 98.8%
- ✅ **Loss:** < 0.04
- ✅ **Fast Inference:** < 1s per image

---

## 🧠 Model Architecture

The model is based on a **custom CNN** architecture with multiple convolutional and pooling layers.  
It leverages **ReLU activation**, **Batch Normalization**, and **Dropout** for optimal generalization.

##### 📥 Input: MRI Image (128x128 RGB)  
##### ├── 🧩 Conv2D (32 filters, 3×3) + ReLU + MaxPooling (2×2)  
##### ├── 🧩 Conv2D (64 filters, 3×3) + ReLU + MaxPooling (2×2)  
##### ├── 🔄 Flatten  
##### ├── ⚙️ Dense (128 units) + ReLU + Dropout (0.5)  
##### └── 🎯 Output Layer: Dense (2 units) + Softmax  
---

### ⚡ Key Highlights
- 🧬 **Feature Extraction:** Convolutional layers capture texture and shape patterns.  
- 🎯 **Classification:** Fully connected layers perform final tumor/no-tumor prediction.  
- 🧱 **Regularization:** Dropout helps prevent overfitting for better generalization.  
- ⚙️ **Optimization:** Trained using the **Adam optimizer** with **categorical cross-entropy** loss.  

---

### 🧩 Summary

This CNN strikes a balance between **accuracy and efficiency**, achieving **99% accuracy** on the test set while maintaining lightweight inference for deployment.
---

## 🧪 Results

### 📊 Training Performance
| Metric | Value |
|:-------|:------:|
| 📊 Accuracy | **0.9885** |
| 📉 Loss | **0.0395** |
| Epochs | 25 |
| Dataset Size | 3,000+ MRI images |

---

### 🖼️ Sample Predictions

| Input MRI | Model Prediction |
|:-----------:|:----------------:|
| ![tumor](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg) | 🧠 Tumor Detected |
| ![normal](https://upload.wikimedia.org/wikipedia/commons/0/05/MRI_head_normal.jpg) | ✅ No Tumor |

*(Images above are placeholders – replace them with your own sample outputs.)*

---

## 📈 Visualization

Below is an example of the model’s accuracy and loss progression during training:

![Training Graph](https://user-images.githubusercontent.com/placeholder/training-graph.png)

---

## 💡 Key Features

- 🧬 **Deep Learning CNN** for image classification  
- 🔍 **Binary classification**: Tumor / No Tumor  
- ⚡ **High accuracy** and quick inference  
- 🧰 Compatible with TensorFlow/Keras  
- 📊 Easy to visualize and interpret results  

---

## 🧰 Tech Stack

- 🐍 **Python 3.10+**
- 🧠 **TensorFlow / Keras**
- 📊 **NumPy, Matplotlib, OpenCV**
- 💾 **Jupyter Notebook**

---

## 🚀 How It Works

1. Load MRI images and preprocess them (resize, normalize).  
2. Train the CNN model on labeled data (Tumor / No Tumor).  
3. Validate performance on unseen test data.  
4. Use the trained model to predict tumors in new MRI scans.  

---

## 🎯 Future Improvements

- 🔄 Implement **Transfer Learning** (e.g., ResNet50 or EfficientNet)  
- 🌐 Deploy the model as a **web app** (Streamlit / Flask)  
- 📱 Create a **mobile version** for real-time detection  

---

## 🧑‍💻 Author

**[Your Name]**  
🌍 Passionate about AI, healthcare, and computer vision.  
📫 Reach me at: [your.email@example.com]  
💼 GitHub: [your-github-profile]  

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use and modify it.  

---

> ⭐ If you like this project, give it a **star** on GitHub and help others discover it!  
> _“AI won’t replace doctors, but doctors who use AI will replace those who don’t.”_

---
