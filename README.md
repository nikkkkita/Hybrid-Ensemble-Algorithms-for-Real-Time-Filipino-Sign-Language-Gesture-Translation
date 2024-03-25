# Hybrid-Ensemble-Algorithms-for-Real-Time-Filipino-Sign-Language-Gesture-Translation

MODEL ARCHITECTURE

This repository contains code for a comprehensive hand gesture recognition system.
Below is a brief overview of the components and their functionalities:

MediaPipe Hands Pretrained Model:
Utilizes MediaPipe Hands library to detect and localize hand keypoints in real-time.
Comprises two pretrained models: Palm Detection Model and Hand Landmarks Detection Model.

Convolutional neural network (CNN) Architecture for Keypoint Classification:

### keypoint_classification.ipynb

Implements a Convolutional Neural Network (CNN) architecture using TensorFlow's Keras API.
Tailored to extract features from keypoint data for hand gesture classification.
Consists of convolutional layers, max-pooling layers, fully connected layers, and dropout regularization.
This code is found inside the simulator and model folder.

Training and Evaluation:

### keypoint_classification.ipynb

Splits the dataset into training and testing sets using sklearn's train_test_split function.
Trains the CNN model on the training data and evaluates its performance on the testing data.
Evaluates accuracy, loss, and generates classification reports and confusion matrices.
This code is found inside the simulator and model folder.

Integration with Lazy Learner Models:
Integrates lazy learner models, namely K-Nearest Neighbors (KNN) and Random Forest, for improved classification.
Fetches data from the CNN's output stored in CSV format for input to lazy learner models.
Demonstrates enhanced accuracy and classification metrics using the combined approach.

### model_testing.py

This is a testing simulator for inference where the skeleton (media pipe framework) is visible and CNN, KNN, and Random Forest prediction is displayed in the console.

load_dataset - Loads the dataset from a CSV file, separating features (X) and labels (y).

find_nearest_classes - Finds the 8 nearest classes for a given data point using KNN, returning the corresponding class labels.

Utilizes RandomForestClassifier from the sklearn.ensemble module for ensemble learning-based classification.
Trains the Random Forest classifier using the preprocessed data and nearest classes.
Uses the trained Random Forest classifier to predict the class and associated probabilities based on the 8 nearest classes.

This code is found inside the model folder.

### keypoint_classification.ipynb

This is a model training script for hand sign recognition. CNN is being integrated to map the sign language.

### model/keypoint_classifier

This directory stores files related to hand sign recognition.<br>
The following files are stored.

- Training data(keypoint.csv)
- Trained model(keypoint_classifier.tflite)
- Label data(keypoint_classifier_label.csv)
- Inference module(keypoint_classifier.py)

### utils/cvfpscalc.py

This is a module for FPS measurement.

Abstract of the Model:
Combines deep learning and lazy learner approaches for hand gesture classification.
Integrates MediaPipe Hands for hand detection, CNN for feature extraction, and lazy learner models for classification.

Achieves high accuracy in recognizing hand gestures, particularly the 26 alphabets in Filipino Sign Language.

FILIPINO SIGN LANGUAGE SIMULATOR

To initiate the Filipino Sign Language simulator, access the 'simulatorfinal.py' file. Upon opening it, the SignLingu window will display. On the homepage, locate the option "Start new Sign Translation" and click on it. This action will prompt a new page to emerge for real-time Sign Language translation. To start the sign language recognition, simply click on the play button. Additionally, within the application, you can explore sections such as "Filipino Sign Language," "Learn more about FSL," "Updates and FAQs," and "About Us."
