import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_image(model_path, img_path, img_size=224):
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    pred = model.predict(img)[0][0]
    class_name = "malignant" if pred > 0.5 else "benign"
    confidence = pred if class_name == "malignant" else 1 - pred
    
    return {
        "class": class_name,
        "confidence": float(confidence),
        "prediction_score": float(pred)
    }

# Example usage
if __name__ == "__main__":
    result = predict_image("models/best_model.h5", "path_to_your_test_image.jpg")
    print(f"Prediction: {result['class']} (Confidence: {result['confidence']:.2%})")