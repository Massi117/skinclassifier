# Import Tensorflow 2.0
import tensorflow as tf 

# Import OpenCV
import cv2 as cv

class DataRetrieval:

    def __init__(self) -> None:
        pass

    def get_dataset(train_directory, val_directory, batch_size, img_height, img_width):

        # Create a training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_directory,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        # Create a validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_directory,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        return train_ds, val_ds


    def rescaleFrame(frame, width, height):
        dimensions = (width,height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)