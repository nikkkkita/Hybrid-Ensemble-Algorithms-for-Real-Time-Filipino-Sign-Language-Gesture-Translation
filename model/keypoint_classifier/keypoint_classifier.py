#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf  # Add this line to import TensorFlow

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
    
        input_data = np.array([landmark_list], dtype=np.float32).reshape(1, 21, 2, 1)
    
        self.interpreter.set_tensor(input_details_tensor_index, input_data)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Softmax function to convert logits to probabilities
        probabilities = tf.nn.softmax(result).numpy()
        
        # Get the index of the class with the highest probability
        result_index = np.argmax(np.squeeze(probabilities))

        # Return the class index and its corresponding probability
        return result_index, probabilities[0, result_index]
