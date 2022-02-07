from tensorflow import keras
from matplotlib import pyplot as plt 
import numpy as np
import onnxruntime

def load_model(model_path):
    session = onnxruntime.InferenceSession(model_path)
    print('Loaded model...')
    return session

def predict_image(image, session):
    
    # compute ONNX Runtime output prediction
    outputs = session.run(None, {'input_1': image})
    mse = np.mean((image - outputs) ** 2)
    if mse <0.002:
        return True
    else:
        return False

    
