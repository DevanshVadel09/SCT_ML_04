# SCT_ML_04

# Hand Gesture Recognition

A machine learning project that recognizes hand gestures using computer vision and deep learning techniques.

## 📁 Project Structure

```
├── app.py                                   # Main application file
├── best_hand_gesture_model.h5               # Trained model weights (best performing)
├── Read Me                                  # Read ME
├── Hand_Gesture_Recognition_model.h5        # Alternative model weights                  
└── hand-gesture-recognition.ipynb           # Original training notebook
```

## 🚀 Features

- Real-time hand gesture recognition
- Pre-trained deep learning models
- Interactive application interface
- Support for multiple gesture classes
- High accuracy gesture classification

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Install required dependencies**
   ```bash
   pip install tensorflow opencv-python numpy matplotlib jupyter
   ```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook (for training notebooks)

## 🎯 Usage

### Running the Application

```bash
python app.py
```

### Training Your Own Model

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook hand-gesture-recognition.ipynb
   ```

2. Follow the notebook cells to:
   - Load and preprocess data
   - Train the model
   - Evaluate performance
   - Save the trained model

## 🤖 Model Information

- **best_hand_gesture_model.h5**: Optimized model with best performance metrics
- **Hand_Gesture_Recognition_model.h5**: Alternative model version

Both models are trained using deep learning techniques and can recognize various hand gestures with high accuracy.

## 📊 Model Performance

The models have been trained and validated to achieve optimal performance in gesture recognition tasks. For detailed performance metrics, refer to the training notebooks.

## 🎮 Supported Gestures

The model can recognize common hand gestures including:
- ✋ Open palm
- ✊ Fist
- 👍 Thumbs up
- ✌️ Peace sign
- And more...

*Note: Specific gesture classes depend on the training data used*

## 🔧 Configuration

You can modify the application behavior by editing the configuration parameters in `app.py`:
- Camera input source
- Model selection
- Confidence threshold
- Display settings

## 📝 Development

### Notebook Files

- **hand-gesture-recognition.ipynb**: Original development notebook
- **hand-gesture-recognition v2.ipynb**: Updated version with improvements

These notebooks contain:
- Data preprocessing steps
- Model architecture definition
- Training process
- Evaluation metrics
- Visualization of results

Data Set Link:-
``` 
https://www.kaggle.com/datasets/gti-upm/leapgestrecog

```

⭐ **Star this repository if you find it helpful!**
