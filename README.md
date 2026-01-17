# Bone-X-ray-image-Classifier-using-DEEP-leaning-model

A Flask-based deep learning web application that classifies bone X-ray images into fracture types using a VGG16 CNN architecture.
The trained model is automatically downloaded from Google Drive at runtime to keep the GitHub repository clean and lightweight.

MODEL DRIVE LINK :
https://drive.google.com/file/d/1sKs5-xIzO4fAWiwqCZYRSAba51bBikPr/view?usp=sharing


eatures

✅ Deep Learning model based on VGG16 (Transfer Learning architecture)

✅ Flask web interface for image upload & prediction

✅ Automatic model download using Google Drive (gdown)

✅ Clean GitHub repo (trained model not stored)

✅ Image preprocessing using preprocess_input

✅ Confidence score shown with prediction

✅ Production-ready project structure

Model Details
Component	Description
Base Model	VGG16 (without top layers)
Input Size	224 × 224 × 3
Architecture	VGG16 → Flatten → Dense(256) → Dense(2)
Output	Softmax
Framework	TensorFlow / Keras
Classes	Oblique fracture, Spiral fracture

Bone-X-ray-image-Classifier-using-DEEP-learning-model/
│
├── app.py                     # Flask application
├── model/
│   └── model_vgg16.h5          # Downloaded automatically (ignored by git)
│
├── static/
│   └── uploads/               # Uploaded images
│
├── templates/
│   └── index.html              # Frontend UI
│
├── .gitignore
├── README.md
└── LICENSE


Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/rafilovestosuffer/Bone-X-ray-image-Classifier-using-DEEP-leaning-model.git
cd Bone-X-ray-image-Classifier-using-DEEP-leaning-model


Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
nstall dependencies
pip install -r requirements.txt


Required packages:

flask
tensorflow
numpy
pillow
gdown

How It Works

User uploads an X-ray image

Image is resized to 224×224

Preprocessed using VGG16 preprocess_input

Model predicts fracture type

Result + confidence score shown on UI

👨‍💻 Author

Rafiur Rahman
Mechanical Engineer | Machine Learning Enthusiast
GitHub: https://github.com/rafilovestosuffer