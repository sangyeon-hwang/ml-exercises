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
    validate_ratio = 0.2
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
    # Training/validation split
    train_As, valid_As, train_Hs, valid_Hs, train_labels, valid_labels = \
        train_test_split(adj_list, feature_list, label_list,
                         test_size=validate_ratio)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_As, train_Hs, train_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((
        valid_As, valid_Hs, valid_labels))
    train_batches = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    train_batches = train_dataset.batch(batch_size=batch_size)
    valid_batches = valid_dataset.batch(batch_size=batch_size)
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
        total_train_loss = 0.
        total_valid_loss = 0.
        train_report = {'accuracies': [],
                        'precisions': [],
                        'recalls': []}
        valid_report = {'accuracies': [],
                        'precisions': [],
                        'recalls': []}

        # Train
        for n_batch, train_batch in enumerate(train_batches):
            A_batch, H_batch, label_batch = train_batch
            loss, pred_batch = train_step(A_batch, H_batch, label_batch)
            total_train_loss += loss
            report = classification_report(label_batch,
                                           np.around(pred_batch),
                                           labels=[0, 1],
                                           zero_division=0,
                                           output_dict=True)
            train_report['accuracies'].append(report['accuracy'])
            train_report['precisions'].append(report['1']['precision'])
            train_report['recalls'].append(report['1']['recall'])

        mean_train_loss = total_train_loss / (n_batch + 1)

        # Validate
        for n_batch, valid_batch in enumerate(valid_batches):
            A_batch, H_batch, label_batch = valid_batch
            loss, pred_batch = evaluate(A_batch, H_batch, label_batch)
            total_valid_loss += loss
            report = classification_report(label_batch,
                                           np.around(pred_batch),
                                           labels=[0, 1],
                                           zero_division=0,
                                           output_dict=True)
            valid_report['accuracies'].append(report['accuracy'])
            valid_report['precisions'].append(report['1']['precision'])
            valid_report['recalls'].append(report['1']['recall'])

        #print(f"Epoch {epoch} batch {n_batch} loss {loss}")

        mean_valid_loss = total_valid_loss / (n_batch + 1)
        mean_train_accuracy = np.mean(train_report['accuracies'])
        mean_train_precision = np.mean(train_report['precisions'])
        mean_train_recall = np.mean(train_report['recalls'])
        mean_valid_accuracy = np.mean(valid_report['accuracies'])
        mean_valid_precision = np.mean(valid_report['precisions'])
        mean_valid_recall = np.mean(valid_report['recalls'])

        print(report_template.format(
            epoch + 1, mean_train_loss,
            mean_train_accuracy, mean_train_precision, mean_train_recall,
            mean_valid_loss,
            mean_valid_accuracy, mean_valid_precision, mean_valid_recall,
            time.time() - start_time
        ))
