# Brain Tumor Classification

This repository contains code for a machine learning model for image classification. The model is trained on brain tumor images and is capable of classifying images into four categories: glioma tumor, meningioma tumor, no tumor, and pituitary tumor.

## Data Preparation

The data for training and testing is stored in the `Data/Training` and `Data/Testing` directories, respectively. The data is loaded, resized, and stored in `X_train` along with their corresponding labels in `y_train`.

## Data Shuffling

The training data is shuffled using the `shuffle` function from the `sklearn` library. This helps in randomizing the data and reducing any bias during training.

## Data Splitting

The shuffled data is split into training and testing sets using the `train_test_split` function from the `sklearn` library. The split ratio is set to 0.1, meaning 10% of the data will be used for testing.

The labels are converted from categorical strings to numerical values using the `labels.index` function. The numerical labels are then converted to one-hot encoded vectors using `tf.keras.utils.to_categorical`.

## Model Building

The model is built using the Keras API. It consists of several convolutional layers with varying filter sizes, activation functions, and pooling layers. Dropout layers are added to prevent overfitting. The output layer consists of 4 neurons with a softmax activation function, representing the 4 possible tumor categories.

The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric. The model summary is displayed, showing the layers, output shapes, and number of parameters.

## Model Training

The model is trained using the `fit` function with the training data (`X_train` and `y_train`) for 30 epochs. The training accuracy, validation accuracy, training loss, and validation loss are recorded during training.

## Accuracy

The current model achieves an accuracy of 89% in classifying brain tumor images. This indicates that the model can correctly classify the majority of the images with a high degree of accuracy.

## Results

The training and validation accuracies are plotted over the epochs to visualize the model's performance. Similarly, the training and validation losses are plotted. The plots are displayed using `matplotlib`.

Please note that the code snippets provided here assume the necessary libraries (`cv2`, `numpy`, `sklearn`, `tensorflow`, `matplotlib`) are imported and relevant functions are defined.
