import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model/keypoint_classifier/keypoint_classifier.hdf5')

# Print the model summary
print(model.summary())

# You can also access individual layers and their configurations
for layer in model.layers:
    print(layer.get_config())
    print(layer.get_weights())  # This will print the layer's weights if it has any
