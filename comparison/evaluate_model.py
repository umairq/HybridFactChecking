from data import Data
import torch
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn


class BaselineEmdeddingsOnlyModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations,dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        # self.loss = torch.nn.BCELoss()

        self.shallom_width = int(12.8 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(num_entities,embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(num_relations,embedding_dim,self.rel_embeddings))})

        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.Dropout(0.20),
                                     nn.ReLU(),
                                     nn.Dropout(0.20),
                                     torch.nn.Linear(self.shallom_width, 1))

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
        return torch.sigmoid(self.shallom(x))


class HybridModelSetting2(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations, dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        # self.loss = torch.nn.BCELoss()
        self.sentence_dim=768*3
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.shallom_width = int(25.6 * self.embedding_dim)
        self.sen_embeddings_train =  dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test
        self.sen_embeddings_valid = dataset.emb_sentences_valid

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)
        self.sentence_embeddings_valid = nn.Embedding(len(self.sen_embeddings_valid), self.sentence_dim)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_entities, embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_relations, embedding_dim, self.rel_embeddings))})

        self.sentence_embeddings_train.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(len(self.sen_embeddings_train), self.sentence_dim, self.sen_embeddings_train))})

        self.sentence_embeddings_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_test), self.sentence_dim, self.sen_embeddings_test))})

        self.sentence_embeddings_valid.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_valid), self.sentence_dim, self.sen_embeddings_valid))})

        self.classification = nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 3 + self.sentence_dim, self.shallom_width),
            nn.BatchNorm1d(self.shallom_width),
            nn.Dropout(.20),
            nn.ReLU(),
            nn.Dropout(.20),
            torch.nn.Linear(self.shallom_width, 1))

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

    def forward(self, e1_idx, rel_idx, e2_idx,sen_idx, type="training"):
        # print(sen_idx)
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        # triplet_embedding = self.shallom(x)
        emb_sen =[]
        if type.__contains__("training"):
            emb_sen = self.sentence_embeddings_train(sen_idx)
        elif type.__contains__("valid"):
            emb_sen = self.sentence_embeddings_valid(sen_idx)
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
        # sentence_embedding = self.shallom_sentence(emb_sen)
        z = torch.cat([emb_head_real, emb_rel_real, emb_tail_real,emb_sen],1)
        z = self.dropout(z)
        return torch.sigmoid(self.classification(z))




datasets_class = ["domain/","domainrange/","mix/","property/","random/","range/"]
cls = datasets_class[0]
method = "emb-only" #emb-only
path_dataset_folder = '../dataset/'
dataset = Data(data_dir=path_dataset_folder, subpath= cls)
num_entities, num_relations = len(dataset.entities), len(dataset.relations)

if method == "emb-only":
    model = BaselineEmdeddingsOnlyModel(embedding_dim=10, num_entities=num_entities, num_relations=num_relations,
                                        dataset=dataset)
else:
    model = HybridModelSetting2(embedding_dim=10, num_entities=num_entities, num_relations=num_relations,
                                dataset=dataset)
model.load_state_dict(torch.load(path_dataset_folder+'models/'+cls.replace("/","-")+'saved_model.pth'))
model.eval()



# Train F1 train dataset
X_train = np.array(dataset.idx_train_data)[:, :3]
y_train = np.array(dataset.idx_train_data)[:, -1]
X_train_tensor = torch.Tensor(X_train).long()

jj = np.arange(0, len(X_train))
x_data = torch.tensor(jj)

    # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
prob = model(idx_s, idx_p, idx_o,x_data).flatten()
pred = (prob > 0.70).float()
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
pred = (prob > 0.70).float()
pred = pred.data.detach().numpy()
print('Acc score on test data', accuracy_score(y_test, pred))
print('report:', classification_report(y_test,pred))
