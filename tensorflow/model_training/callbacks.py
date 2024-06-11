import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load your dataset and preprocess it
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Define your model architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Define your callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=3)
checkpoint = ModelCheckpoint(
    "model_checkpoint.h5", monitor="val_accuracy", save_best_only=True
)

# Train the model with callbacks
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[early_stopping, checkpoint],
)
