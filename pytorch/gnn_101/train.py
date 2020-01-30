import time
import warnings

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
import torch_geometric

import dataset
import gnn
import utils

if __name__ == '__main__':
    train_csv_path = '../../data/qed_classification/train.csv'
    test_csv_path = '../../data/qed_classification/test.csv'
    num_train_data = 1000
    num_test_data = 200
    batch_size = 32
    num_convs = 3
    hidden_dim = 128
    p_drop = 0.
    num_epochs = 20
    lr = 1e-3

    # Suppress sklearn warning
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')

    # Data loading
    train_dataset = dataset.MoleculeDataset(train_csv_path,
                                            num_data=num_train_data)
    train_loader = torch_geometric.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = dataset.MoleculeDataset(test_csv_path,
                                           num_data=num_test_data)
    test_loader = torch_geometric.data.DataLoader(
        test_dataset, batch_size=batch_size)
    print("Num of batches:", len(train_loader))

    model = gnn.GNN(train_dataset.num_node_features,
                    hidden_dim,
                    num_convs,
                    p_drop).to(device)
    print("Num of parameters:",
          sum(param.numel() for param in model.parameters()
              if param.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_obj = torch.nn.BCEWithLogitsLoss()

    title_template = "  ".join(["{:5}"] * 10)
    report_template = "  ".join(["{:5}"] + ["{:.3f}"]*9)
    print(f"Training starts! Total {num_epochs} epochs")
    print(title_template.format("Epoch", "Train", "", "", "",
                                "Valid", "", "", "", "Time"))
    print(title_template.format("", "Loss", "Acc", "Prec", "Rec",
                                "Loss", "Acc", "Prec", "Rec", ""))

    for epoch in range(num_epochs):
        start_time = time.time()
        train_losses = []
        test_losses = []
        train_report = {'accuracies': [],
                        'precisions': [],
                        'recalls': []}
        valid_report = {'accuracies': [],
                        'precisions': [],
                        'recalls': []}

        # Train
        model.train()
        for i_batch, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(batch.to(device))  # (num_batch, 1)
            loss = loss_obj(out, batch.y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().numpy())

            pred = torch.sigmoid(out.detach()).cpu().numpy()
            report = classification_report(batch.y.cpu(),
                                           np.around(pred),
                                           labels=[0, 1],
                                           zero_division=0,
                                           output_dict=True)
            train_report['accuracies'].append(report['accuracy'])
            train_report['precisions'].append(report['1']['precision'])
            train_report['recalls'].append(report['1']['recall'])

        # Validate
        model.eval()
        for i_batch, batch in enumerate(test_loader):
            with torch.no_grad():
                out = model(batch.to(device))
            loss = loss_obj(out, batch.y)
            test_losses.append(loss.data.cpu().numpy())

            pred = torch.sigmoid(out.detach()).cpu().numpy()
            report = classification_report(batch.y.cpu(),
                                           np.around(pred),
                                           labels=[0, 1],
                                           zero_division=0,
                                           output_dict=True)
            valid_report['accuracies'].append(report['accuracy'])
            valid_report['precisions'].append(report['1']['precision'])
            valid_report['recalls'].append(report['1']['recall'])

        mean_train_loss = np.mean(train_losses)
        mean_test_loss = np.mean(test_losses)
        mean_train_accuracy = np.mean(train_report['accuracies'])
        mean_train_precision = np.mean(train_report['precisions'])
        mean_train_recall = np.mean(train_report['recalls'])
        mean_valid_accuracy = np.mean(valid_report['accuracies'])
        mean_valid_precision = np.mean(valid_report['precisions'])
        mean_valid_recall = np.mean(valid_report['recalls'])

        print(report_template.format(
            epoch + 1, mean_train_loss,
            mean_train_accuracy, mean_train_precision, mean_train_recall,
            mean_test_loss,
            mean_valid_accuracy, mean_valid_precision, mean_valid_recall,
            time.time() - start_time
        ))
