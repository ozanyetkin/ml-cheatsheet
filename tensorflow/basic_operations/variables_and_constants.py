import tensorflow as tf

# Create a constant tensor
a = tf.constant(5)
b = tf.constant(3)

# Perform basic operations
addition = tf.add(a, b)
subtraction = tf.subtract(a, b)
multiplication = tf.multiply(a, b)
division = tf.divide(a, b)

# Create a variable tensor
c = tf.Variable(2)

# Update the variable tensor
update_c = tf.assign(c, tf.add(c, 1))

# Initialize all variables
init = tf.global_variables_initializer()

# Run the TensorFlow session
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Perform basic operations
    result_addition = sess.run(addition)
    result_subtraction = sess.run(subtraction)
    result_multiplication = sess.run(multiplication)
    result_division = sess.run(division)

    # Print the results
    print("Addition:", result_addition)
    print("Subtraction:", result_subtraction)
    print("Multiplication:", result_multiplication)
    print("Division:", result_division)

    # Update and print the variable tensor
    sess.run(update_c)
    result_c = sess.run(c)
    print("Updated c:", result_c)
