

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

(trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.fashion_mnist.load_data()
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

trainImages = trainImages / 255.
testImages = testImages / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=trainImages.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(trainImages, trainLabels, epochs=5)
testLoss, testAccuracy = model.evaluate(testImages, testLabels)
print("Test loss:", testLoss)
print("Test accuracy:", testAccuracy)

# `model.predict` accepts a *batch* of data points,
# so an additional dimension has to be added.
print(model.predict(testImages[0][None,...]))
