from data import Data
from sklearn.metrics import auc
import numpy as np

# datasets_class = ["date/","domain/","domainrange/","mix/","property/","random/","range/"]
#
# path_dataset_folder = 'dataset/'
# dataset = Data(data_dir=path_dataset_folder, subpath= datasets_class[1])

from data import Data
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# extracting evedience sentence and generating training and testing triples from the list of facts in factbecnh.
class Baseline1_FactCheck:
    def __init__(self, data_dir=None,multiclass=True):
        # self.data = []
        if multiclass:
            datasets_class = ["date/", "domain/", "domainrange/", "mix/", "property/", "random/", "range/"]
            for sub_path in datasets_class:
                self.extract_and_save_result_of_factcheck_output_multiclass(self, data_dir, sub_path)
        else:
             self.extract_and_save_result_of_factcheck_output(self, data_dir)


    @staticmethod
    def extract_and_save_result_of_factcheck_output(self, data_dir):
        self.dataset = Data(data_dir=data_dir+"complete_dataset/")
        print("test")
        self.save_data(self,data_dir+"complete_dataset/")




    @staticmethod
    def extract_and_save_result_of_factcheck_output_multiclass(self, data_dir, multiclass):
        self.dataset = Data(data_dir=data_dir, subpath=multiclass)
        print("test")
        self.save_data(self,data_dir,multiclass)

    @staticmethod
    def save_data(self,data_dir="",multiclass="" ):
        # saving the ground truth values
        with open(data_dir+"data/test/"+multiclass+"ground_truth_test.nt", "w") as prediction_file:
            new_line = "\n"
            # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.test_data)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
            prediction_file.write(new_line)
        with open(data_dir+"data/train/"+multiclass+"ground_truth_train.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.train_set)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"

            prediction_file.write(new_line)

        # saving the pridiction values
        with open(data_dir+"data/test/"+multiclass+"prediction_test_pred.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.test_data_pred)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
            prediction_file.write(new_line)
        with open(data_dir+"data/train/"+multiclass+"prediction_train_pred.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.train_set_pred)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"

            prediction_file.write(new_line)






path_dataset_folder = '../dataset/'
se = Baseline1_FactCheck(data_dir=path_dataset_folder,multiclass=True)








#

#


# dx = 5
# xx = np.arange(1,100,dx)
# yy = np.arange(1,100,dx)
#
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(xx,yy)))
# print('computed AUC using np.trapz: {}'.format(np.trapz(yy, dx = dx)))
