import tensorflow as tf
import time

# Size of the matrix (adjustable)
MATRIX_SIZE = 4096

# Create random matrices
def make_matrices():
    return tf.random.normal([MATRIX_SIZE, MATRIX_SIZE]), tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])

# Time execution
def benchmark(device_name):
    with tf.device(device_name):
        a, b = make_matrices()
        # Warm-up
        tf.matmul(a, b)

        start = time.time()
        for _ in range(10):
            tf.matmul(a, b)
        tf.keras.backend.clear_session()
        end = time.time()

    return end - start

# Run benchmarks
cpu_time = benchmark("/CPU:0")
print(f"CPU time: {cpu_time:.2f} seconds")

if tf.config.list_physical_devices('GPU'):
    gpu_time = benchmark("/GPU:0")
    print(f"GPU time: {gpu_time:.2f} seconds")
else:
    print("No GPU found.")
