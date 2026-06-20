import json, os
import numpy as np
from tensorflow import keras
from src.config import MODEL_PATH, CLASS_JSON_PATH, IMG_SIZE
from src.preprocessing import ALL_CUSTOM_OBJECTS

def load_model():
    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH, custom_objects=ALL_CUSTOM_OBJECTS)
    print("Model loaded!")
    return model

def load_class_names():
    if os.path.exists(CLASS_JSON_PATH):
        with open(CLASS_JSON_PATH) as f:
            return json.load(f)
    return ["lion","tiger","leopard","elephant","camel","bear",
            "rabbit","mouse","squirrel","dolphin","whale","shark",
            "snake","lizard","crocodile","bus","pickup_truck","train",
            "bicycle","motorcycle","apple","orange","pear",
            "rose","tulip","sunflower","skyscraper","house",
            "rocket","mushroom"]

def predict(model, class_names, pil_image):
    from PIL import Image
    img   = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr   = np.expand_dims(np.array(img, dtype="float32"), 0)
    probs = model.predict(arr, verbose=0)[0]
    top5  = np.argsort(probs)[::-1][:5]
    results    = {class_names[i]: float(probs[i]) for i in top5}
    top1_class = class_names[top5[0]]
    top1_conf  = float(probs[top5[0]]) * 100
    return results, f"**Prediction:** **{top1_class.capitalize()}** ({round(top1_conf, 2)}%)"
