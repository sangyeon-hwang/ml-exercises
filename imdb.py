
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

imdb = tf.keras.datasets.imdb
vocabularySize = 10000
(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words=vocabularySize)
    # trainData  : (25000,)-ndarray[list[int]]
    # trainLabels: (25000,)-ndarray[int64]
    # testData   : (25000,)-ndarray[list[int]]
    # testLabels : (25000,)-ndarray[int64]

wordIdxs = imdb.get_word_index()
wordDict = {idx+3:word for word,idx in wordIdxs.items()}
wordDict[0] = "<PAD>"
wordDict[1] = "<START>"
wordDict[2] = "<UNK>"  # unknown
wordDict[3] = "<UNUSED>"

def decode(idxs: list) -> str:
    return ' '.join(wordDict.get(idx, '?') for idx in idxs)

## Padding to len 256
trainData = tf.keras.preprocessing.sequence.pad_sequences(trainData,
                                                          value=0,                                                                                            padding='post',
                                                          maxlen=256)
testData = tf.keras.preprocessing.sequence.pad_sequences(testData,
                                                         value=0,                                                                                            padding='post',
                                                         maxlen=256)
    ## Both become (25000, 256)-ndarray[int32].

## Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocabularySize, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## Data splitting
validateData = trainData[:10000]
validateLabels = trainLabels[:10000]
subTrainData = trainData[10000:]
subTrainLabels = trainLabels[10000:]

## Train.
history = model.fit(subTrainData, subTrainLabels,
                    epochs=40,
                    batch_size=512,
                    validation_data=(validateData, validateLabels),
                    verbose=1)

## Evaluate.
results = model.evaluate(testData, testLabels)
print("Test loss, accuracy =", results)

## Plot the history.
historyDict = history.history
epochs = range(1, 1 + 40)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(epochs, historyDict['loss'], 'bo', label='Training loss')
plt.plot(epochs, historyDict['val_loss'], 'b', label='Validation loss')
plt.title('Training and valiation losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, historyDict['accuracy'], 'ro', label='Training accuracy')
plt.plot(epochs, historyDict['val_accuracy'], 'r', label='Validation accuracy')
plt.title('Training and valiation accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
