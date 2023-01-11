import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as GNN
from torch_geometric.data import DataLoader
import torch_geometric.transforms as transform
from torch_geometric.datasets import ShapeNet
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning import losses
from sklearn.manifold import TSNE
import seaborn as sns

class network(nn.Module):
        def __init__(self, k=20, aggr='max'):
            super(network, self).__init__()
            # Feature extraction
            self.augmentation = transform.Compose(
                [transform.RandomJitter(0.03), transform.RandomFlip(1), transform.RandomShear(0.2)])
            self.conv1 = GNN.DynamicEdgeConv(GNN.MLP([2 * 3, 64, 64]), k, aggr)
            self.conv2 = GNN.DynamicEdgeConv(GNN.MLP([2 * 64, 128]), k, aggr)

            # Encoder head
            self.lin1 = nn.Linear(128 + 64, 128)
            # Projection head (See explanation in SimCLRv2)
            self.mlp = GNN.MLP([128, 256, 32], norm=None)


        def forward(self, data, train=True):

            if train:
                # Get 2 augmentations of the batch
                augm_1 = self.augmentation(data)
                augm_2 = self.augmentation(data)

                # Extract properties
                pos_1, batch_1 = augm_1.pos, augm_1.batch
                pos_2, batch_2 = augm_2.pos, augm_2.batch

                # Get representations for first augmented view
                x1 = self.conv1(pos_1, batch_1)
                x2 = self.conv2(x1, batch_1)
                h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

                # Get representations for second augmented view
                x1 = self.conv1(pos_2, batch_2)
                x2 = self.conv2(x1, batch_2)
                h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))

                # Global representation
                h_1 = GNN.global_max_pool(h_points_1, batch_1)
                h_2 = GNN.global_max_pool(h_points_2, batch_2)
            else:
                x1 = self.conv1(data.pos, data.batch)
                x2 = self.conv2(x1, data.batch)
                h_points = self.lin1(torch.cat([x1, x2], dim=1))
                return GNN.global_max_pool(h_points, data.batch)

            # Transformation for loss function
            compact_h_1 = self.mlp(h_1)
            compact_h_2 = self.mlp(h_2)
            return h_1, h_2, compact_h_1, compact_h_2


def plot_point_cloud(X):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='g', s=3, marker='o', alpha=0.6)

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)

    ax.set_xlabel('R', fontsize=10)
    ax.set_ylabel('G', fontsize=10)
    ax.set_zlabel('B', fontsize=10)

    ax.axis('off')

    plt.show()

def train(model, optimizer, train_data, criterion, device, epoch):
    loss = 0
    for index, batch in enumerate(tqdm(train_data)):
        batch = batch.to(device)
        h_1, h_2, compact_h_1, compact_h_2 = model(batch)
        label_ = torch.arange(0, compact_h_1.shape[0])
        label = torch.cat((label_, label_))
        loss_ = criterion(torch.cat((compact_h_1, compact_h_2), dim=0), label)
        # visualize(compact_h_1, batch.category)
        loss += loss_
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    return loss
    # if epoch % 100 == 0:
    #     print(f'loss at epoch {epoch} is equal to {loss}')

def train_fn():
    dataset_ = ShapeNet(root='.', categories=["Table", "Lamp", "Guitar", "Motorbike"])
    data_loader = DataLoader(dataset_, batch_size=32, shuffle=True)
    model = network()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = losses.NTXentLoss()
    # scheduler=optim.lr_scheduler.ExponentialLR()
    loss_arr = []
    for epoch in range(20):
        loss = train(model, optimizer, data_loader, criterion, device, epoch)
        loss_arr.append(loss)
    data_loader = DataLoader(dataset_, batch_size=1000, shuffle=True)
    batch = next(iter(data_loader))
    h_1, h_2, compact_h_1, compact_h_2 = model(batch)
    visualize(compact_h_1, batch.category)
    losses_float = [float(loss.cpu().detach().numpy()) for loss in loss_arr]
    plt_ = sns.lineplot(losses_float)
    plt_.set(xlabel='epoch', ylabel='error')
    plt.savefig('train.png')

def visualize(out, color):
    fig = plt.figure(figsize=(5, 5), frameon=False)
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    color = color.detach().cpu().numpy()
    plt.scatter(z[:, 0],
                z[:, 1],
                s=10,
                c=color,
                cmap="Set2"
                )

    plt.legend()
    plt.savefig('tsne.png')
    # fig.canvas.draw()
if __name__=='__main__':
    train_fn()

    #
    # X = next(iter(data_loader))
    # model = network()
    # out = model(X)

# def train():
#     model.train()
#     total_loss = 0
#     for _, data in enumerate(tqdm.tqdm(data_loader)):
#         data = data.to(device)
#         optimizer.zero_grad()
#         # Get data representations
#         h_1, h_2, compact_h_1, compact_h_2 = model(data)
#         # Prepare for loss
#         embeddings = torch.cat((compact_h_1, compact_h_2))
#         # The same index corresponds to a positive pair
#         indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
#         labels = torch.cat((indices, indices))
#         loss = loss_func(embeddings, labels)
#         loss.backward()
#         total_loss += loss.item() * data.num_graphs
#         optimizer.step()
#     return total_loss / len(dataset)
#
# for epoch in range(1, 4):
#     loss = train()
#     print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
#     scheduler.step()