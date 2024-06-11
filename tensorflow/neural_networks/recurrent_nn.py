import tensorflow as tf


# Define the RNN model
class RNNModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(RNNModel, self).__init__()
        self.hidden_units = hidden_units
        self.rnn_cell = tf.keras.layers.SimpleRNNCell(self.hidden_units)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_state = self.rnn_cell.get_initial_state(
            batch_size=batch_size, dtype=tf.float32
        )
        outputs, _ = tf.nn.dynamic_rnn(
            self.rnn_cell, inputs, initial_state=hidden_state
        )
        logits = self.dense(outputs[:, -1, :])
        return logits


# Create an instance of the RNN model
rnn_model = RNNModel(hidden_units=64)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Define the training loop
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = rnn_model(inputs)
        loss_value = loss_fn(labels, logits)
    grads = tape.gradient(loss_value, rnn_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, rnn_model.trainable_variables))
    return loss_value


# Generate some dummy data for training
inputs = tf.random.normal(shape=(32, 10, 32))
labels = tf.random.uniform(shape=(32, 1), minval=0, maxval=2, dtype=tf.int32)

# Train the RNN model
for epoch in range(10):
    loss = train_step(inputs, labels)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
