from data import Data
from sklearn.metrics import f1_score
import torch
from torch.nn.init import xavier_normal_
import torch.nn as nn
from numpy.random import RandomState
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics


class Baseline(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()

        self.shallom_width = int(2 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                     torch.nn.Linear(self.shallom_width, 1))

    def forward(self, e1_idx, rel_idx, e2_idx):
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        return torch.sigmoid(self.shallom(x))
class Baseline2(torch.nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.loss = torch.nn.BCELoss()

        # input_dim embeddings of triples and embeddings of textual info
        self.fc=torch.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        s represents vector representation of triple and embeddings of textual evidence
        :param x:
        :return:
        """
        # several layers of affine trans.
        return torch.sigmoid(pred)

datasets_class = ["date/","domain/","domainrange/","mix/","property/","random/","range/"]

path_dataset_folder = 'dataset/'
dataset = Data(data_dir=path_dataset_folder, subpath= datasets_class[5])
num_entities, num_relations = len(dataset.entities), len(dataset.relations)
model = Baseline(embedding_dim=10, num_entities=num_entities, num_relations=num_relations)

X_dataloader = DataLoader(torch.Tensor(dataset.idx_train_data).long(), batch_size=256, num_workers=4, shuffle=False)

optimizer = torch.optim.Adam(model.parameters())
for i in range(500):
    loss_of_epoch = 0
    for fact_mini_batch in X_dataloader:
        # 1. Zero the gradient buffers
        optimizer.zero_grad()
        idx_s, idx_p, idx_o, label = fact_mini_batch[:, 0], fact_mini_batch[:, 1], fact_mini_batch[:,
                                                                                   2], fact_mini_batch[:, 3]
        # Label conversion
        label = label.float()
        # 2. Forward
        # [e_i, r_j, e_k, 1] => indexes of entities and relation.
        # Emb of tripes concat emb of textual embeddings
        pred = model(idx_s, idx_p, idx_o).flatten()
        # 3. Compute Loss
        loss = model.loss(pred, label)
        loss_of_epoch += loss.item()
        # 4. Backprop loss
        loss.backward()
        # 6. Update weights with respect to loss.
        optimizer.step()

    # if i % 100 == 0:
        print('Loss:', loss_of_epoch)
model.eval()


# Train F1 train dataset
X_train = np.array(dataset.idx_train_data)[:, :3]
y_train = np.array(dataset.idx_train_data)[:, -1]
X_train_tensor = torch.Tensor(X_train).long()
idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
prob = model(idx_s, idx_p, idx_o).flatten()
pred = (prob > 0.6).float()
pred = pred.data.detach().numpy()
print('Acc score on train data', accuracy_score(y_train, pred))

# Train F1 test dataset
X_test = np.array(dataset.idx_test_data)[:, :3]
y_test = np.array(dataset.idx_test_data)[:, -1]
X_test_tensor = torch.Tensor(X_test).long()
idx_s, idx_p, idx_o = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2]
prob = model(idx_s, idx_p, idx_o).flatten()
pred = (prob > 0.6).float()
pred = pred.data.detach().numpy()
print('Acc score on test data', accuracy_score(y_test, pred))
