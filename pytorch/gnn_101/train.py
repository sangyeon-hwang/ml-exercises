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
    max_num_atoms = None

    # Suppress sklearn warning
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')

    # Data loading
    max_num_atoms = None
    train_dataset = dataset.MoleculeDataset(train_csv_path,
                                            num_train_data,
                                            max_num_atoms)
    test_dataset = dataset.MoleculeDataset(test_csv_path,
                                           num_test_data,
                                           max_num_atoms)
    if max_num_atoms is None:
        loader = torch_geometric.data.DataLoader
    else:
        loader = torch_geometric.data.DenseDataLoader
    train_loader = loader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = loader(test_dataset, batch_size=batch_size)

    model = gnn.GCN(train_dataset.num_node_features,
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
        train_loss = 0.
        test_loss = 0.
        train_preds, train_labels = [], []
        test_preds, test_labels = [], []

        # Train
        model.train()
        for i_batch, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(batch.to(device))  # (num_batch, 1)
            loss = loss_obj(out, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.sigmoid(out.detach())
            train_preds.extend(pred.tolist())
            train_labels.extend(batch.y.squeeze().tolist())

        train_loss /= i_batch + 1
        train_report = classification_report(train_labels,
                                             np.around(train_preds),
                                             labels=[0, 1],
                                             zero_division=0,
                                             output_dict=True)

        # Validate
        model.eval()
        for i_batch, batch in enumerate(test_loader):
            with torch.no_grad():
                out = model(batch.to(device))
            loss = loss_obj(out, batch.y)

            test_loss += loss.item()
            pred = torch.sigmoid(out.detach())
            test_preds.extend(pred.tolist())
            test_labels.extend(batch.y.squeeze().tolist())

        test_loss /= i_batch + 1
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
