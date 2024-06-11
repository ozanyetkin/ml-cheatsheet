import tensorflow as tf

# Create two tensors
tensor1 = tf.constant([1, 2, 3])
tensor2 = tf.constant([4, 5, 6])

# Perform element-wise addition
addition = tf.add(tensor1, tensor2)

# Perform element-wise multiplication
multiplication = tf.multiply(tensor1, tensor2)

# Perform matrix multiplication
matrix_multiplication = tf.matmul(
    tf.reshape(tensor1, [1, 3]), tf.reshape(tensor2, [3, 1])
)

# Print the results
with tf.Session() as sess:
    print("Tensor1:", sess.run(tensor1))
    print("Tensor2:", sess.run(tensor2))
    print("Addition:", sess.run(addition))
    print("Multiplication:", sess.run(multiplication))
    print("Matrix Multiplication:", sess.run(matrix_multiplication))
