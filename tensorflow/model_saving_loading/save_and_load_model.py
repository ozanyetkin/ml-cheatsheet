import tensorflow as tf

# Define x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define your model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
model.fit(x_train, y_train, epochs=10)

# Save the model
model.save("/path/to/save/model")

# Load the model
loaded_model = tf.keras.models.load_model("/path/to/save/model")

# Evaluate the loaded model
loss, accuracy = loaded_model.evaluate(x_test, y_test)
print("Restored model accuracy: {:5.2f}%".format(100 * accuracy))
