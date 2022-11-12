# image_viewer.py
import io
import os
import PySimpleGUI as sg
from PIL import Image
import cv2 as cv

# NN imports
import tensorflow as tf
from nn_model.model import NN_model as nn
from nn_model.DataRetrieval import DataRetrieval as dd

sg.theme('Default')

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]
def main():
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
            sg.Button("Predict"),
        ],
        [sg.Text(size=(40, 1), key="-TOUT-")],
    ]
    window = sg.Window("Melinoma Detection", layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
        if event == "Predict":
            directory = values["-FILE-"]
            if os.path.exists(directory):
                # Define the model
                new_model = nn.build_model()
                # Initialize the model by passing some data through
                new_model.build(input_shape=(1,64,64,3))

                new_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam', metrics = ['accuracy'])

                # Loads the weights
                new_model.load_weights("nn_model/detection.h5")

                # Load the image
                img = cv.imread(directory)
                # Preprocess the image
                img_resized = dd.rescaleFrame(img, 64, 64)
                new_img = tf.convert_to_tensor(img_resized, dtype=tf.float32)
                img_tensor = tf.image.convert_image_dtype(new_img, dtype=tf.float32, saturate=False)
                img_tensor = tf.expand_dims(img_tensor, axis=0)

                # Predict each frame
                prediction = new_model.predict(img_tensor)
                if prediction[0][0] >= prediction[0][1]:
                    txt = 'Single Image Prediction: Benign'
                else:
                    txt = 'Single Image Prediction: Malignant'
                
                window["-TOUT-"].update(txt)

    window.close()
if __name__ == "__main__":
    main()