import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
import os

save_path = "uploaded_files"

if not os.path.exists(save_path):
    os.makedirs(save_path)

seed_value = 42  # Replace with any integer value
tf.random.set_seed(seed_value)

def dicom_to_jpeg(dicom_file):
    """Converts a DICOM file to a JPEG image."""
    try:
        dicom = pydicom.dcmread(dicom_file)
        pixel_array = dicom.pixel_array
        pixel_array_scaled = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        image = Image.fromarray(pixel_array_scaled)
        return image
    except Exception as e:
        st.error(f"Error converting DICOM to JPEG: {e}")
        return None

def load_model1(model_path):
    """Loads a Keras model."""
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_class(model, image, class_labels):
    """Predicts the class of an image."""
    try:
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label
    except Exception as e:
        st.error(f"Error predicting class: {e}")
        return None

st.title("Predição de Densidade Mamária")

uploaded_file = st.file_uploader("Selecione o arquivo", type=["dcm", "jpeg", "jpg", "png"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()


    file_path = os.path.join(save_path, uploaded_file.name)

    if file_extension == "dcm":
        image = dicom_to_jpeg(uploaded_file)
        output_filename = uploaded_file.name.replace('.dcm', '.jpg')
        file_path = os.path.join(save_path, output_filename)
        
        image.save(file_path, "JPEG")

    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    image = cv2.imread(file_path)
    preprocessed_image = cv2.resize(image, (128, 128))


    st.image(image, caption='Imagem selecionada', use_column_width=True)

    model_paths = {
        'densidade': 'new_breast_densityII_128_10k.h5',      
        'calcificações': 'calcificacoes.h5',
        'distorção arquitetural': 'distorcao.h5',
        'espessamento, retração de pele ou areolar': 'espessamento.h5',
        'esteatonecrose': 'esteatonecrose.h5',
        'linfonodo intramamário': 'linfonodo.h5',
        'nódulo': 'nodulo.h5'
    }

    class_labels_achados = ["Não", "Sim"]

    for model_name, model_path in model_paths.items():
        model = load_model1(model_path)

        if model:
            if model_name == 'densidade':
                # Define your list of class labels
                class_labels_density = ["BIRADS A", "BIRADS B", "BIRADS C", "BIRADS D"]
                prediction_density = model.predict(np.expand_dims(preprocessed_image, axis=0))
                predicted_class_index_density = np.argmax(prediction_density)
                predicted_class_label_density = class_labels_density[predicted_class_index_density]
                
                if predicted_class_label_density in ["BIRADS A", "BIRADS B"]:
                    st.write(f"Mama não densa: {predicted_class_label_density}")
                else:
                    st.write(f"Mama densa: {predicted_class_label_density}")
            else:
                predicted_class_label = predict_class(model, preprocessed_image, class_labels_achados)
                if predicted_class_label:
                    st.write(f"{model_name.capitalize()}: {predicted_class_label}")
