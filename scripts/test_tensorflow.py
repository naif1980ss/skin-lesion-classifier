import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("List physical devices:", tf.config.list_physical_devices())
mps = any('MPS' in str(d) for d in tf.config.list_physical_devices())
print("MPS available:", mps)
