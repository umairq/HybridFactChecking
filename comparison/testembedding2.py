from data import Data
import torch.nn as nn
from numpy.random import RandomState
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random
import numpy as np
# from pytorchtools import EarlyStopping
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



class BaselineEmdeddingsOnlyModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations,dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        # self.loss = torch.nn.BCELoss()

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(num_entities,embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(num_relations,embedding_dim,self.rel_embeddings))})


        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
                                     nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width, self.shallom_width2),
                                     nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width2, self.shallom_width3),
                                     nn.Dropout(0.50),
                                     nn.BatchNorm1d(self.shallom_width3),
                                     # # nn.Dropout(0.50),
                                     nn.ReLU(self.shallom_width3),
                                     # # nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width3, 1))

    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        for i, word in enumerate(embeddings):
            try:
                weights_matrix[i] = word
                words_found += 1
            except KeyError:
                print('test')
                exit(1)
        return weights_matrix

    def forward(self, e1_idx, rel_idx, e2_idx,sen_idx="", type="training"):
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        x2 = self.shallom(x)
        x3 = torch.sigmoid(x2)
        return x3



# "date/",
datasets_class = ["domain/","domainrange/","mix/","random/","property/","range/"]
for cls in datasets_class:
    method = "emb-only" #emb-only
    path_dataset_folder = '../dataset/'
    dataset = Data(data_dir=path_dataset_folder, subpath= cls)
    num_entities, num_relations = len(dataset.entities), len(dataset.relations)
    if method == "emb-only":
        model = BaselineEmdeddingsOnlyModel(embedding_dim=10, num_entities=num_entities, num_relations=num_relations, dataset=dataset)
    # else:
        # model = HybridModelSetting2(embedding_dim=10, num_entities=num_entities, num_relations=num_relations, dataset=dataset)

    bat_size = int(len(dataset.idx_train_data)/3)+1
    bat_size_valid = int(len(dataset.idx_valid_data) / 3) + 1
    X_dataloader = DataLoader(torch.Tensor(dataset.idx_train_data).long(), batch_size=bat_size, num_workers=0, shuffle=True)
    X_valid_dataloader = DataLoader(torch.Tensor(dataset.idx_valid_data).long(), batch_size=bat_size_valid, num_workers=0, shuffle=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # accuracy                           0.63      1135

    optimizer = torch.optim.Adam(model.parameters())
    # Declaring Criterion and Optimizer  BCE is good for classification
    criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    min_valid_loss = np.inf
    # test_jj, valid_jj = generate_data(dataset.test_data, dataset.valid_set)
    for i in range(1500):
        jj = []
        iter=0
        loss_of_epoch = 0
        model.train()
        for fact_mini_batch in X_dataloader:

            idx_s, idx_p, idx_o, label = fact_mini_batch[:, 0], fact_mini_batch[:, 1], fact_mini_batch[:,
                                                                                       2], fact_mini_batch[:, 3]
            # Label conversion
            label = label.float()


            jj = np.arange(iter, iter + len(idx_s))
            iter += len(idx_s)
            x_data = torch.tensor(jj)

            # 2. Forward pass
            pred = model(idx_s, idx_p, idx_o,x_data,"training").flatten()


            # 3. Compute Loss
            loss = criterion(pred, label)
            # Replaces pow(2.0) with abs() for L1 regularization

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())

            loss = loss + l2_lambda * l2_norm

            # 1. Zero the gradient buffers
            optimizer.zero_grad()
            # 4. Backprop loss  Calculate gradients
            loss.backward()
            # 6. Update weights with respect to loss.
            optimizer.step()
            # Calculate Loss
            loss_of_epoch += loss.item()

        # if i % 100 == 0:
        print(f'Epoch {i + 1} \t\t Training Loss: {loss_of_epoch / len(X_dataloader)}')
        model.eval()
        iter2 = 0
        valid_loss = 0.0
        for valid_mini_batch in X_valid_dataloader:

            idx_s, idx_p, idx_o, label = valid_mini_batch[:, 0], valid_mini_batch[:, 1], valid_mini_batch[:,
                                                                                       2], valid_mini_batch[:, 3]
            # Label conversion
            label = label.float()

            jj = np.arange(iter2, iter2 + len(idx_s))
            iter2 += len(idx_s)
            x_data = torch.tensor(jj)

            pred = model(idx_s, idx_p, idx_o, x_data, "valid").flatten()
            # Find the Loss
            loss = criterion(pred, label)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())

            loss = loss + l2_lambda * l2_norm
            # Calculate Loss
            valid_loss += loss.item()


        print(f'Epoch {i + 1} \t\t Training Loss: {loss_of_epoch / len(X_dataloader)} \t\t Validation Loss: {valid_loss / len(X_valid_dataloader)}')
        if  i%5000==0 and min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            # if i%100==0:
            torch.save(model.state_dict(), path_dataset_folder+'models/'+cls.replace("/","-")+'saved_model.pth')

    # Train F1 train dataset
    X_train = np.array(dataset.idx_train_data)[:, :3]
    y_train = np.array(dataset.idx_train_data)[:, -1]
    X_train_tensor = torch.Tensor(X_train).long()

    jj = np.arange(0, len(X_train))
    x_data = torch.tensor(jj)

    # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
    idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
    prob = model(idx_s, idx_p, idx_o,x_data).flatten()
    pred = (prob > 0.6).float()
    pred = pred.data.detach().numpy()
    print('Acc score on train data', accuracy_score(y_train, pred))
    print('report:', classification_report(y_train,pred))

    # Train F1 test dataset
    X_test = np.array(dataset.idx_test_data)[:, :3]
    y_test = np.array(dataset.idx_test_data)[:, -1]
    X_test_tensor = torch.Tensor(X_test).long()
    idx_s, idx_p, idx_o = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2]

    jj = np.arange(0, len(X_test))
    x_data = torch.tensor(jj)

    prob = model(idx_s, idx_p, idx_o,x_data,"testing").flatten()
    pred = (prob > 0.6).float()
    pred = pred.data.detach().numpy()
    print('Acc score on test data', accuracy_score(y_test, pred))
    print('report:', classification_report(y_test,pred))
    exit(1)
    torch.save(model.state_dict(), path_dataset_folder+'data/train/'+cls+'model-'+method+'-15000.pth')

