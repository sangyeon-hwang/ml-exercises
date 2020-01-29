import torch
import torch.nn.functional as F
import torch_geometric

import dataset
import gnn
import utils

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')
    train_csv_path = '../data/qed_train_data.csv'
    train_dataset = dataset.MoleculeDataset(train_csv_path)
    #train_loader = torch_geometric.data.DataLoader(
    #    train_dataset, batch_size=16, shuffle=True)

    model = gnn.GNN(train_dataset.num_node_features, 16).to(device)
    print("Num of parameters:",
          sum(param.numel() for param in model.parameters()
              if param.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_obj = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        for i in range(10):
            data = train_dataset[i].to(device)
            out = model(data)
            loss = loss_obj(out, data.y)
            print(loss); exit()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss {loss}")

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(data))
    print("pred:", pred)
    print("label", data.y)
