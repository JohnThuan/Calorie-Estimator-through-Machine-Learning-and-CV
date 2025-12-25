# Calorie Estimation through Computer Vision

A mobile application that estimates calorie content from food images using deep learning. Built with React Native (Expo) and FastAPI, featuring a custom-trained EfficientNet-B4 model on the Food-101 dataset.

## Screenshots

  <p align="center">
    <img src="screenshots/home.png" width="250" alt="Home Screen" />
    <img src="screenshots/camera1.png" width="250" alt="Camera Screen" />
    <img src="screenshots/camera2.png" width="250" alt="Camera" />
    <img src="screenshots/steak.png" width="250" alt="Results Screen" />
    <img src="screenshots/icecream.png" width="250" alt="Results Screen" />
  </p>



## Features

- **Real-time Food Recognition**: Capture or upload food images for instant analysis
- **Custom ML Model**: EfficientNet-B4 trained on Food-101 dataset (86% validation accuracy)
- **Nutritional Breakdown**: Detailed macronutrient information (protein, carbs, fats)
- **Portion Size Estimation**: Computer vision-based portion size analysis
- **Baseline Comparison**: Compare detected items against a reference baseline
- **Analysis History**: Track and review past meal analyses


## Technical Stack

### Mobile App
- **Framework**: React Native with Expo SDK 54
- **Language**: TypeScript
- **Navigation**: React Navigation
- **Camera**: Expo Camera & Image Picker
- **UI**: Custom professional theme with data-focused interface

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **ML Framework**: PyTorch
- **Model**: EfficientNet-B4 (19M parameters)
- **Dataset**: Food-101 (101,000 images, 101 classes)
- **Image Processing**: PIL, OpenCV
- **API**: RESTful architecture

## Model Performance

- **Architecture**: EfficientNet-B4
- **Training Data**: Food-101 dataset (75,750 training images)
- **Validation Accuracy**: 86.06%
- **Parameters**: 19 million
- **Image Size**: 224x224
- **Training Techniques**:
  - Transfer learning with ImageNet pre-training
  - Data augmentation (RandAugment, TrivialAugmentWide)
  - Label smoothing (0.1)
  - Dropout (0.4)
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate schedule
  - Early stopping


## Project Structure

```
CalAi/
├── mobile/                 # React Native mobile app
│   ├── src/
│   │   ├── screens/       # Camera, Results, History screens
│   │   ├── services/      # API integration
│   │   ├── components/    # Reusable UI components
│   │   └── styles/        # Theme and styling
│   ├── App.tsx
│   └── package.json
│
└── backend/               # FastAPI backend
    ├── services/
    │   ├── food_recognizer_efficientnet_b4.py
    │   ├── calorie_calculator.py
    │   ├── image_processor.py
    │   └── food101_nutrition.py
    ├── train_model_efficientnet_b4.py
    ├── download_dataset.py
    ├── main.py
    └── requirements.txt
```

## Setup Instructions

### Backend Setup

1. **Install Python dependencies**:
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

2. **Download and prepare dataset** (for training):
```bash
pip install -r requirements-ml.txt
python download_dataset.py
```

3. **Train the model** (optional - pre-trained model included):
```bash
python train_model_efficientnet_b4.py
```

4. **Start the server**:
```bash
python main.py
```
Server runs on `http://0.0.0.0:8000`

### Mobile App Setup

1. **Install dependencies**:
```bash
cd mobile
npm install
```

2. **Update API endpoint**:
Edit `mobile/src/services/api.ts` and set your backend IP:
```typescript
const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';
```

3. **Start the app**:
```bash
npm start
```

4. **Run on device**:
- Scan QR code with Expo Go app (iOS/Android)
- Or press `w` for web version

## API Endpoints

- `POST /analyze` - Analyze food image and return calorie estimate
- `POST /baseline/save` - Save baseline reference image
- `POST /baseline/compare` - Compare current image with baseline
- `GET /history` - Retrieve analysis history
- `GET /health` - Health check endpoint

## Model Training

The model was trained on an NVIDIA RTX 4070 GPU with the following configuration:

- **Batch Size**: 48
- **Epochs**: 25 (with early stopping)
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.01)
- **Loss**: CrossEntropyLoss with label smoothing
- **Training Time**: ~4 hours

To retrain the model:
```bash
cd backend
python train_model_efficientnet_b4.py
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=runs/
```

## GPU Requirements

- Training: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- Inference: Can run on CPU or GPU
- CUDA 12.1+ and cuDNN required for GPU acceleration

## Nutritional Database

The application uses a comprehensive nutritional database covering all 101 Food-101 categories with:
- Calories per 100g
- Protein, carbohydrates, and fat content
- Portion size adjustments based on image analysis

## Future Improvements

- Multi-food detection in single image
- More granular portion size estimation
- Expanded food category support
- User-specific dietary recommendations
- Meal planning integration

## License

MIT License

## Acknowledgments

- Food-101 dataset by ETH Zurich
- EfficientNet architecture by Google Research
- PyTorch and torchvision teams

