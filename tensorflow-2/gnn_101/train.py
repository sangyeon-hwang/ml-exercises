import time
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

import data_process

class Attn(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(dim, activation='relu')

    def call(self, A, H):
        return self.dense(tf.linalg.matmul(A, H))

class MyModel(tf.keras.Model):
    def __init__(self, dim, num_attn):
        super().__init__()
        self.blocks = [Attn(dim) for _ in range(num_attn)]
        self.final = tf.keras.layers.Dense(1)

    def readout(self, H):
        R = tf.reduce_mean(H, axis=1)
        return R

    def call(self, A, H0):
        H = H0
        for attn in self.blocks:
            H = attn(A, H)
        # H: (batch, num_atoms, num_features)
        R = self.readout(H)
        out = self.final(R)
        return out

def train_step(A_batch, H_batch, label_batch):
    with tf.GradientTape() as tape:
        output = model(A_batch, H_batch)  # (batch_size, 1)
        output = tf.squeeze(output)  # (batch_size,)
        loss = loss_func(label_batch, output)

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    pred_batch = tf.sigmoid(output).numpy()
    return loss, pred_batch

def evaluate(A_batch, H_batch, label_batch):
    output = model(A_batch, H_batch)  # (batch_size, 1)
    output = tf.squeeze(output)  # (batch_size,)
    loss = loss_func(label_batch, output)
    pred_batch = tf.sigmoid(output).numpy()
    return loss, pred_batch

if __name__ == '__main__':
    # Suppress sklearn warning
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    # Parameters
    num_data = 1000
    test_ratio = 0.2
    max_num_atoms = 200
    batch_size = 32
    shuffle_buffer_size = 1000
    dense_dim = 128
    num_attn = 3
    num_epochs = 20

    # Load and process the data
    start_time = time.time()
    adj_list, feature_list, label_list = data_process.load_data(
        '../../data/qed_classification/train.csv',
        max_num_atoms,
        num_data)
    # Training/test split
    train_As, test_As, train_Hs, test_Hs, train_labels, test_labels = \
        train_test_split(adj_list, feature_list, label_list,
                         test_size=test_ratio)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_As, train_Hs, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((
        test_As, test_Hs, test_labels))
    train_batches = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    train_batches = train_dataset.batch(batch_size=batch_size)
    test_batches = test_dataset.batch(batch_size=batch_size)
    print(f"Loading finished, {time.time() - start_time:.3f} sec")

    optimizer=tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model = MyModel(dim=dense_dim, num_attn=num_attn)

    # Training loop
    title_template = "  ".join(["{:5}"] * 10)
    report_template = "  ".join(["{:5}"] + ["{:.3f}"]*9)

    print(f"Training starts! Total {num_epochs} epochs")
    print(title_template.format("Epoch", "Train", "", "", "",
                                "Valid", "", "", "", "Time"))
    print(title_template.format("", "Loss", "Acc", "Prec", "Rec",
                                "Loss", "Acc", "Prec", "Rec", ""))

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.
        test_loss = 0.
        train_preds, train_labels = [], []
        test_preds, test_labels = [], []

        # Train
        for n_batch, train_batch in enumerate(train_batches):
            A_batch, H_batch, label_batch = train_batch
            loss, pred_batch = train_step(A_batch, H_batch, label_batch)

            train_loss += loss.numpy()
            train_preds.extend(pred_batch)
            train_labels.extend(label_batch)

        train_loss /= n_batch + 1
        train_report = classification_report(train_labels,
                                             np.around(train_preds),
                                             labels=[0, 1],
                                             zero_division=0,
                                             output_dict=True)

        # Validate
        for n_batch, test_batch in enumerate(test_batches):
            A_batch, H_batch, label_batch = test_batch
            loss, pred_batch = evaluate(A_batch, H_batch, label_batch)

            test_loss += loss.numpy()
            test_preds.extend(pred_batch)
            test_labels.extend(label_batch)

        test_loss /= n_batch + 1
        test_report = classification_report(test_labels,
                                            np.around(test_preds),
                                            labels=[0, 1],
                                            zero_division=0,
                                            output_dict=True)

        print(report_template.format(
            epoch + 1,
            train_loss, train_report['accuracy'],
            train_report['1']['precision'], train_report['1']['recall'],
            test_loss, test_report['accuracy'],
            test_report['1']['precision'], test_report['1']['recall'],
            time.time() - start_time
        ))
