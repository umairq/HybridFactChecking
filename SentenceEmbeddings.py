from data import Data
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# extracting evedience sentence and generating embeddings from those evedence sentences and storing them in CSV file format
class SentenceEmbeddings:
    def __init__(self, data_dir=None):
        self.extract_sentence_embeddings_from_factcheck_output(data_dir)






    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output(data_dir=None):
        data_train = []
        data_test = []
        pred = []
        test = False
        train = False
        correct = False
        print(data_dir)
        dataset1 = Data(data_dir=data_dir)
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
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    if o == "" : o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    # print("line:" + line+ ":"+ score + ":"+ str(correct))
                    sentences = []
                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            p2 = p1.split(", proofPhrase=")[1]
                            print(str(idx) + ":" + p2)
                            sentences.append(p2)

                    if test == True and train == False:
                        print("test" + str(correct))
                        data_test.append([s, p, o, correct, sentences])

                    if test == False and train == True:
                        print("train" + str(correct))
                        data_train.append([s, p, o, correct, sentences])

        model = SentenceTransformer('nq-distilbert-base-v1')
        embeddings_textual_evedences_train = []
        embeddings_textual_evedences_test = []
        for idx, (s, p, o, c, p2) in enumerate(data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            for s2 in p2:
                print("\n" + s2)
            avg_embedding = np.mean(sentence_embeddings, axis=0)

            if (np.isnan(np.sum(avg_embedding))):
                avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train.append(avg_embedding)




        for idx, (s, p, o, c, p2) in enumerate(data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            for s2 in p2:
                print("\n" + s2)
            avg_embedding = np.mean(sentence_embeddings, axis=0)

            print(avg_embedding)
            if (np.isnan(np.sum(avg_embedding))):
                avg_embedding = np.zeros(768, dtype=int)
                print(avg_embedding)
                # exit(1)

            embeddings_textual_evedences_test.append(avg_embedding)



        X = np.array(embeddings_textual_evedences_train)
        print(X.shape)
        X=pd.DataFrame(X)
        compression_opts = dict(method='zip',archive_name='trainSE.csv')
        X.to_csv(data_dir+'trainSE.zip', index=False,compression=compression_opts)

        Y = np.array(embeddings_textual_evedences_test)
        print(Y.shape)
        Y=pd.DataFrame(Y)
        compression_opts1 = dict(method='zip',archive_name='testSE.csv')
        Y.to_csv(data_dir+'testSE.zip', index=False,compression=compression_opts1)



path_dataset_folder = 'dataset/'
se = SentenceEmbeddings(data_dir=path_dataset_folder)
