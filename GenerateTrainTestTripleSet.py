from data import Data
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# extracting evedience sentence and generating training and testing triples from the list of facts in factbecnh.
class GenerateTrainTestTriplesSet:
    def __init__(self, data_dir=None):
        self.extract_sentence_embeddings_from_factcheck_output(data_dir)

    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output(data_dir):
        data_train = []
        data_test = []
        pred = []
        test = False
        train = False
        correct = False

        with open(data_dir+"textResults.txt", "r") as file1:
            for line in file1:
                if line.__contains__("factbench/factbench/test/"):
                    test = True
                    train = False
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    continue
                if line.__contains__("factbench/factbench/train/"):
                    test = False
                    train = True
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    continue

                if line.startswith(' defactoScore'):
                    l1 = line.split("defactoScore: ")[1]
                    score = l1.split(" setProofSentences")[0]
                    x = line.split("subject : ")[1]
                    so = x.split(" object : ")
                    s = so[0].replace(" ", "_")
                    print(s)
                    assert s != ""
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    print(o)
                    if o == "": o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    print(p)
                    assert p != ""
                    print("line:" + line + ":" + score + ":" + str(correct))

                    if test == True and train == False:
                        print("test")
                        data_test.append([s, p, o, correct])

                    if test == False and train == True:
                        print("train")
                        data_train.append([s, p, o, correct])

        with open(data_dir+ "train.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, lbl) in enumerate(data_train):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

            prediction_file.write(new_line)

        with open(data_dir+"test.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, lbl) in enumerate(data_test):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

            prediction_file.write(new_line)



        @staticmethod
        def get_relations(data):
            relations = sorted(list(set([d[1] for d in data])))
            return relations

        @staticmethod
        def get_entities(data):
            entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
            return entities



path_dataset_folder = 'dataset/'
se = GenerateTrainTestTriplesSet(data_dir=path_dataset_folder)
