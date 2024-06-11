import tensorflow as tf

# Define x_train, y_train, x_val, y_val, x_test, y_test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define your model
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

# Create a checkpoint callback
checkpoint_path = "/path/to/save/checkpoint.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

# Train the model with the checkpoint callback
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[checkpoint_callback],
)

# Load the saved checkpoint
model.load_weights(checkpoint_path)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
