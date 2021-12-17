from data import Data
# from embedding_only_approach import Baseline, Baseline3
import torch.nn as nn

import random
import numpy as np
import torch



# from embedding_only_approach import Baseline and hybrid models

class BaselineEmdeddingsOnlyModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations,dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()

        self.shallom_width = int(12.8 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(num_entities,embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(num_relations,embedding_dim,self.rel_embeddings))})

        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
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



class HybridModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations, dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()
        self.sentence_dim=768*3

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.sen_embeddings_train =  dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_entities, embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_relations, embedding_dim, self.rel_embeddings))})

        self.sentence_embeddings_train.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(len(self.sen_embeddings_train), self.sentence_dim, self.sen_embeddings_train))})

        self.sentence_embeddings_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_test), self.sentence_dim, self.sen_embeddings_test))})


        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                     torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.shallom_sentence = nn.Sequential(torch.nn.Linear(self.sentence_dim , self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                     torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.classification = nn.Sequential(torch.nn.Linear(self.shallom_width * 2, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
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
        triplet_embedding = self.shallom(x)
        emb_sen =[]
        if type.__contains__("training"):
            emb_sen = self.sentence_embeddings_train(sen_idx)
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
        sentence_embedding = self.shallom_sentence(emb_sen)
        z = torch.cat([triplet_embedding,sentence_embedding],1)
        return torch.sigmoid(self.classification(z))


class HybridModelSetting2(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations, dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()
        self.sentence_dim=768*3

        self.shallom_width = int(25.6 * self.embedding_dim+self.sentence_dim)
        self.sen_embeddings_train =  dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_entities, embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_relations, embedding_dim, self.rel_embeddings))})

        self.sentence_embeddings_train.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(len(self.sen_embeddings_train), self.sentence_dim, self.sen_embeddings_train))})

        self.sentence_embeddings_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_test), self.sentence_dim, self.sen_embeddings_test))})


        self.classification = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3+self.sentence_dim , self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
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
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
        # sentence_embedding = self.shallom_sentence(emb_sen)
        z = torch.cat([emb_head_real, emb_rel_real, emb_tail_real,emb_sen],1)
        return torch.sigmoid(self.classification(z))

def save_data(dataset, data_dir="", multiclass="", training="test",  scores=[], method="hybrid"):
    # saving the ground truth values
    with open(data_dir + "data/"+training+ "/" + multiclass + "ground_truth_"+training+ "_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
        for idx, (head, relation, tail, score) in enumerate(
                (dataset.test_data)):
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + \
                        head.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + \
                        relation.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + \
                        tail.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(
                score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)


    # saving the pridiction values
    with open(data_dir + "data/"+training+ "/" + multiclass + "prediction_"+training+ "_pred_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        for idx, (tuple, score) in enumerate(
                zip(dataset.test_data,scores)):
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + \
                        tuple[0].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + \
                        tuple[1].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + \
                        tuple[2].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(
                float(score)) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)


# "date/",

datasets_class = ["domain/","domainrange/","mix/","property/","random/","range/"]
# cls = 6
for cls in datasets_class:
    path_dataset_folder = '../dataset/'
    method = "hybrid"   # emb-only or hybrid
    dataset = Data(data_dir=path_dataset_folder, subpath= cls)
    num_entities, num_relations = len(dataset.entities), len(dataset.relations)
    if method=="emb-only":
        model = BaselineEmdeddingsOnlyModel(embedding_dim=10, num_entities=num_entities, num_relations=num_relations, dataset=dataset)
    else:
        model = HybridModelSetting2(embedding_dim=10, num_entities=num_entities, num_relations=num_relations, dataset=dataset)
    # loading the trained model
    model.load_state_dict(torch.load(path_dataset_folder+'data/train/'+cls+'model-'+method+'-1500.pth'))
    model.eval()


    # train data
    jj = np.arange(0, len(dataset.idx_train_data))
    x_data = torch.tensor(jj)
    X_train_tensor = torch.Tensor(dataset.idx_train_data).long()
    idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
    prob = model(idx_s, idx_p, idx_o,x_data).flatten()
    save_data(dataset, path_dataset_folder,cls, "train",scores=prob,method=method)

    # testing data
    jj = np.arange(0, len(dataset.idx_test_data))
    x_data = torch.tensor(jj)
    X_train_tensor = torch.Tensor(dataset.idx_test_data).long()
    idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
    prob = model(idx_s, idx_p, idx_o,x_data, "testing").flatten()
    save_data(dataset, path_dataset_folder,cls, "test",scores=prob,method=method)



# for s,p,o, lbl in dataset.idx_test_data:
#     print("test")
