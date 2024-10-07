# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inceptionv3_preprocess,
)
from tqdm.keras import TqdmCallback
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier

# Global variables
IMG_SIZE = 160
BATCH_SIZE = 32
FILE_PATH = "/Users/jai/Project_India/data/cotton_plant_disease"
CLASSES = os.listdir(FILE_PATH)[1:]
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 50
LEARNING_RATE = 0.0001

# Step 1: Load and Preprocess the Dataset
def load_data():
    """Load the image dataset from directory."""
    image_train = image_dataset_from_directory(
        FILE_PATH,
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        class_names=CLASSES,
        subset="training",
    )
    image_valid = image_dataset_from_directory(
        FILE_PATH,
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        class_names=CLASSES,
        subset="validation",
    )

    return image_train.prefetch(buffer_size=AUTOTUNE), image_valid.prefetch(
        buffer_size=AUTOTUNE
    )


# Step 2: Data Augmentation
def data_augmentation_layer():
    """Apply data augmentation to the training images."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.2),
            layers.RandomZoom(0.5, 0.2),
        ]
    )


# Step 3: Base Model - Using Pre-trained Models
def build_base_model(model_name="InceptionV3"):
    """Return a pre-trained model without the top classification layer."""
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    if model_name == "MobileNetV2":
        base_model = MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = mobilenet_preprocess
    elif model_name == "VGG16":
        base_model = VGG16(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = vgg_preprocess
    elif model_name == "ResNet50":
        base_model = ResNet50(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = resnet_preprocess

    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = efficientnet_preprocess
    elif model_name == "InceptionV3":
        base_model = InceptionV3(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = inceptionv3_preprocess
    base_model.trainable = False
    return base_model, preprocess_input


# Step 4: Add Classification Head
def add_classification_head(base_model, preprocess_input, data_augmentation):
    """Attach classification head to the pre-trained base model."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


# Step 5: Compile and Train Model
def compile_and_train(model, ds_train, ds_valid, epochs=EPOCHS):
    """Compile and train the model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, verbose=0
    )

    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=epochs,
        callbacks=[TqdmCallback(verbose=0), earlyStopping],
        verbose=0,
    )
    return history


# Step 6: Fine-tuning Model
def fine_tune_model(
    base_model, model, ds_train, ds_valid, lr=LEARNING_RATE / 10, fine_tune_at=100
):
    """Unfreeze the top layers and fine-tune the model."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, verbose=0
    )
    history_finetune = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=EPOCHS,
        callbacks=[TqdmCallback(verbose=0), earlyStopping],
        verbose=0,
    )

    return history_finetune


# Step 7: Ensemble Predictions
def ensemble_predictions(models, input_image):
    """Average the predictions from multiple models."""
    preds = [model.predict(input_image) for model in models]
    avg_pred = np.mean(preds, axis=0)
    return avg_pred


# Step 8: Main Function
def main():
    ds_train, ds_valid = load_data()
    data_augmentation = data_augmentation_layer()
    pretrained_list = ["EfficientNetB0"]
    base_model_list, models_list = [], []
    model_name = "InceptionV3"
    # Build a base model
    base_model, preprocess_input = build_base_model(model_name)
    model = add_classification_head(base_model, preprocess_input, data_augmentation)
    hist = compile_and_train(model, ds_train, ds_valid)
    train_acc = "{:.2%}".format(hist.history["accuracy"][-1])
    val_acc = "{:.2%}".format(hist.history["val_accuracy"][-1])
    print(
        f"Training Accuracy: {train_acc}, Validation Accuracy: {val_acc} Before Fine Tuning"
    )
    # Trying different hyper parameters
    lrn_rates = [1e-5]
    for lrn_rate in lrn_rates:
        history_finetune = fine_tune_model(
            base_model, model, ds_train, ds_valid, fine_tune_at=100, lr=1e-5
        )
        train_acc = "{:.2%}".format(history_finetune.history["accuracy"][-1])
        val_acc = "{:.2%}".format(history_finetune.history["val_accuracy"][-1])
        print(
            f"Training Accuracy: {train_acc}, Validation Accuracy: {val_acc} after fine tuning at learning rate {lrn_rate}"
        )
        models_list.append(model)
        base_model_list.append(base_model)


if __name__ == "__main__":
    main()
