# from torch import hub
from sentence_transformers import SentenceTransformer

from data import Data
import numpy as np
import pandas as pd
# from sentence_transformers import SentenceTransformer
from select_top_n_sentences import select_top_n_sentences
import zipfile

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# extracting evedience sentence and generating embeddings from those evedence sentences and storing them in CSV file format
class SentenceEmbeddings:
    def __init__(self, data_dir=None,multiclass=True):
        self.data_file = "textResult6.txt"
        if multiclass:
            self.extract_sentence_embeddings_from_factcheck_output_multiclass(self, data_dir)
        else:
            self.extract_sentence_embeddings_from_factcheck_output(self,data_dir)



    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output(self,data_dir=None):
        data_train = []
        data_test = []
        pred = []
        test = False
        train = False
        correct = False
        print(data_dir)
        # dataset1 = Data(data_dir=data_dir)
        with open(data_dir+self.data_file, "r") as file1:
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

        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings_textual_evedences_train = []
        embeddings_textual_evedences_test = []
        for idx, (s, p, o, c, p2,t1) in enumerate(data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            for s2 in p2:
                print("\n" + s2)
            avg_embedding = np.mean(sentence_embeddings, axis=0)

            if (np.isnan(np.sum(avg_embedding))):
                avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train.append(avg_embedding)

        triple_emb = model.encode(s + " " + p + " " + o)
        for idx, (s, p, o, c, p2,t1) in enumerate(data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                if cosine(triple_emb, st1) < 0.1:
                    p2.remove(ss)
                else:
                    print(cosine(triple_emb, st1))

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
        with zipfile.ZipFile(data_dir+'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        Y = np.array(embeddings_textual_evedences_test)
        print(Y.shape)
        Y=pd.DataFrame(Y)
        compression_opts1 = dict(method='zip',archive_name='testSE.csv')
        Y.to_csv(data_dir+'testSE.zip', index=False,compression=compression_opts1)
        with zipfile.ZipFile(data_dir+'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    @staticmethod
    def entityToNLrepresentation(self, predicate):
        p = predicate
        if p == "birthPlace":
            p = "born in"
        if p == "deathPlace":
            p = "died in"
        if p == "foundationPlace":
            p = "founded in"
        if p == "starring":
            p = "starring in"
        if p == "award":
            p = "awarded with"
        if p == "subsidiary":
            p = "subsidiary of"
        if p == "author":
            p = "author of"
        if p == "spouse":
            p = "spouse of"
        if p == "office":
            p = "office"
        if p == "team":
            p = "team"
        return p

    # path of the training or testing folder
    # data2 contains data
    # type2 contains either test or train string
    @staticmethod
    def saveSentenceEmbeddingToFile(self, path, data2 , type2,n):
        X = np.array(data2)
        print(X.shape)
        X = X.reshape(X.shape[0], 768*n)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name=type2+'SE.csv')
        X.to_csv(path + type2+'SE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +type2+ 'SE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

    @staticmethod
    def getSentenceEmbeddings(self,data_dir, data, model, true_statements_embeddings, type = 'default', n =3):
        embeddings_textual_evedences = []
        website_ids = set()
        for idx, (s, p, o, c, p2, t1) in enumerate(data):
            # p = self.entityToNLrepresentation(self, p)
            print('triple->'+s+' '+p+' '+o)
            # triple_emb = model.encode(s + " " + p + " " + o.replace("_", " "))
            # print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            temp = []
            temp2 = []
            for tt in enumerate(p3):
                temp.append(tt[1].split(",website=")[0])
                temp2.append((tt[1].split(",website=")[1]).split("/")[-1])
            # for idx2, ss in enumerate(p3):
            #     st1 = model.encode(ss)
            #     # choose the best sentence
            #     if len(temp) == 0:
            #         temp.append(ss)
            #     test2 = temp.pop()
            #     test = model.encode(test2)
            #     if cosine(triple_emb, st1) > cosine(triple_emb, test):
            #         temp.append(ss)
            #     else:
            #         temp.append(test2)
            #     #         ///////////////////////////////////
            #     if cosine(triple_emb, st1) < 0.1:
            #         p2.remove(ss)
            #     else:
            #         print(cosine(triple_emb, st1))
            # # Sentences are encoded by calling model.encode()
            for item in temp2:
                website_ids.add(item)

            sentence_embeddings =[]
            if (s+' '+p+' '+o) not in true_statements_embeddings.keys():
                temp22, temp33 = select_top_n_sentences(temp2, n, temp,type)
                sentence_embeddings = model.encode(temp33)
                true_statements_embeddings[s+' '+p+' '+o] = sentence_embeddings
            else:
                sentence_embeddings = true_statements_embeddings[s+' '+p+' '+o]

            # for sen in temp33:
            #     e1 = mode

            # for s2 in p2:
            #     print("" + s2)
            # for s2 in temp:
            #     print("" + s2)

            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            # print(sentence_embeddings.shape)
            if (np.size(sentence_embeddings) == 0):
                sentence_embeddings = np.zeros((n, 768), dtype=int)

            if (np.size(sentence_embeddings) == 768*(n-2)):
                sentence_embeddings = np.append(sentence_embeddings,(np.zeros((n-1, 768), dtype=int)), axis=0)

            if (np.size(sentence_embeddings) == 768*(n-1)):
                sentence_embeddings = np.append(sentence_embeddings,(np.zeros((n-2, 768), dtype=int)), axis=0)

            embeddings_textual_evedences.append(sentence_embeddings)
        # with open('/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/pg_ranks/' + 'all_websites_ids_'+type+'.txt',"w") as f:
        #     for item in list(website_ids):
        #         f.write("%s\n" % item)
        path = ""
        if type.__contains__("test"):
            path = data_dir + "data/test/" + type.replace("_test","") + "/"
            self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
        else:
            path = data_dir + "data/train/" + type + "/"
            self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "train", n)
        return embeddings_textual_evedences, true_statements_embeddings

    # path = data_dir + "data/test/range/"
    # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_range, "test")
    # @staticmethod
    # def getSentenceEmbeddings(self, data, model):
    #     embeddings_textual_evedences = []
    #     for idx, (s, p, o, c, p2,t1) in enumerate(data):
    #         p = self.entityToNLrepresentation(self,p)
    #
    #         triple_emb = model.encode(s + " " + p + " " + o.replace("_"," "))
    #         print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
    #         p3 = p2.copy()
    #         temp = []
    #         for idx2, ss in enumerate(p3):
    #             st1 = model.encode(ss)
    #             # choose the best sentence
    #             if len(temp) == 0:
    #                 temp.append(ss)
    #             test2 = temp.pop()
    #             test = model.encode(test2)
    #             if cosine(triple_emb, st1) > cosine(triple_emb, test):
    #                 temp.append(ss)
    #             else:
    #                 temp.append(test2)
    #             #         ///////////////////////////////////
    #             if cosine(triple_emb, st1) < 0.1:
    #                 p2.remove(ss)
    #             else:
    #                 print(cosine(triple_emb, st1))
    #         # Sentences are encoded by calling model.encode()
    #
    #         sentence_embeddings = model.encode(temp)
    #
    #         for s2 in p2:
    #             print("" + s2)
    #         for s2 in temp:
    #             print("" + s2)
    #
    #         # avg_embedding = np.mean(sentence_embeddings, axis=0)
    #         print(sentence_embeddings.shape)
    #         if (np.size(sentence_embeddings)==0):
    #             sentence_embeddings = np.zeros((1,768), dtype=int)
    #
    #         embeddings_textual_evedences.append(sentence_embeddings)
    #
    #     return embeddings_textual_evedences

    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_multiclass(self,data_dir=None):
        data_train = []
        data_test = []

        date_data_train = []
        date_data_test = []
        domain_data_train = []
        domain_data_test = []
        domainrange_data_train = []
        domainrange_data_test = []
        mix_data_train = []
        mix_data_test = []
        property_data_train = []
        property_data_test = []
        random_data_train = []
        random_data_test = []
        range_data_train = []
        range_data_test = []
        multiclass_neg_exp = True

        pred = []
        test = False
        train = False
        correct = False
        print(data_dir)
        # dataset1 = Data(data_dir=data_dir)
        with open(data_dir+self.data_file, "r") as file1:
            for line in file1:
                if line.__contains__("factbench/factbench/test/"):
                    test = True
                    train = False
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True

                    neg_data_dir = "true"
                    if multiclass_neg_exp:
                        if line.__contains__("/test/wrong/date"):
                            neg_data_dir = "wrong/date/"
                        if line.__contains__("/test/wrong/domain"):
                            neg_data_dir = "wrong/domain/"
                        if line.__contains__("/test/wrong/domainrange"):
                            neg_data_dir = "wrong/domainrange/"
                        if line.__contains__("/test/wrong/mix"):
                            neg_data_dir = "wrong/mix/"
                        if line.__contains__("/test/wrong/property"):
                            neg_data_dir = "wrong/property/"
                        if line.__contains__("/test/wrong/random"):
                            neg_data_dir = "wrong/random/"
                        if line.__contains__("/test/wrong/range"):
                            neg_data_dir = "wrong/range/"

                    continue
                if line.__contains__("factbench/factbench/train/"):
                    test = False
                    train = True
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True

                    neg_data_dir = "true"
                    if multiclass_neg_exp:
                        if line.__contains__("/train/wrong/date"):
                            neg_data_dir = "wrong/date/"
                        if line.__contains__("/train/wrong/domain"):
                            neg_data_dir = "wrong/domain/"
                        if line.__contains__("/train/wrong/domainrange"):
                            neg_data_dir = "wrong/domainrange/"
                        if line.__contains__("/train/wrong/mix"):
                            neg_data_dir = "wrong/mix/"
                        if line.__contains__("/train/wrong/property"):
                            neg_data_dir = "wrong/property/"
                        if line.__contains__("/train/wrong/random"):
                            neg_data_dir = "wrong/random/"
                        if line.__contains__("/train/wrong/range"):
                            neg_data_dir = "wrong/range/"

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
                    websites = []
                    trustworthiness = []
                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            website = proof.split("website='")[1].split("', proofPhrase")[0].replace(" ","_")
                            p2 = p1.split(", proofPhrase='")[1]
                            p3 = p2.split("', trustworthinessScore='")
                            print(str(idx) + ":" + p3[0] + "tr: " +p3[1][:-1])
                            sentences.append(p3[0] + ",website="+website)
                            websites.append(website)
                            trustworthiness.append(p3[1][:-1])

                    if test == True and train == False:
                        print("test" + str(correct))
                        data_test.append([s, p, o, correct, sentences,trustworthiness])

                    if test == False and train == True:
                        print("train" + str(correct))
                        data_train.append([s, p, o, correct, sentences,trustworthiness])

                    if multiclass_neg_exp == True:
                        if test == False and train == True:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_train.append([s, p, o, correct, sentences,trustworthiness])

                        if test == True and train == False:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_test.append([s, p, o, correct, sentences,trustworthiness])


        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        # model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/2")

        # embeddings_textual_evedences_train = []
        # embeddings_textual_evedences_test = []
        #
        # embeddings_textual_evedences_train_date = []
        # embeddings_textual_evedences_train_domain = []
        # embeddings_textual_evedences_train_domainrange = []
        # embeddings_textual_evedences_train_mix = []
        # embeddings_textual_evedences_train_property = []
        # embeddings_textual_evedences_train_random = []
        # embeddings_textual_evedences_train_range = []
        true_statements_embeddings = {}
        n = 3
        # ///////////////////////////////////////////////////////////////////////////////train
        embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir,date_data_train,model,true_statements_embeddings, 'date',n)

        # exit(1)
        # for idx, (s, p, o, c, p2,t1) in enumerate(date_data_train):
        #     p = self.entityToNLrepresentation(p)
        #
        #     triple_emb = model.encode(s + " " + p + " " + o.replace("_"," ").replace(",",""))
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     temp = []
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         # choose the best sentence
        #         if len(temp) == 0:
        #             temp.append(ss)
        #         test2 = temp.pop()
        #         test = model.encode(test2)
        #         if cosine(triple_emb, st1) > cosine(triple_emb, test):
        #             temp.append(ss)
        #         else:
        #             temp.append(test2)
        #         #         ///////////////////////////////////
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #     # Sentences are encoded by calling model.encode()
        #
        #     sentence_embeddings = model.encode(temp)
        #
        #     for s2 in p2:
        #         print("" + s2)
        #     for s2 in temp:
        #         print("" + s2)
        #
        #     # avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(sentence_embeddings))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_date.append(sentence_embeddings)
        embeddings_textual_evedences_train_domain, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir,domain_data_train, model,true_statements_embeddings, 'domain',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(domain_data_train):
        #     p = self.entityToNLrepresentation(p)
        #
        #     triple_emb = model.encode(s + " " + p + " " + o.replace("_"," ").replace(",",""))
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_domain.append(avg_embedding)
        embeddings_textual_evedences_train_domainrange, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, domainrange_data_train, model,true_statements_embeddings, 'domainrange',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(domainrange_data_train):
        #     p = self.entityToNLrepresentation(p)
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_domainrange.append(avg_embedding)
        embeddings_textual_evedences_train_mix, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, mix_data_train, model,true_statements_embeddings, 'mix',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(mix_data_train):
        #     p = self.entityToNLrepresentation(p)
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #     # # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_mix.append(avg_embedding)
        embeddings_textual_evedences_train_property, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, property_data_train, model,true_statements_embeddings, 'property',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(property_data_train):
        #     p = self.entityToNLrepresentation(p)
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_property.append(avg_embedding)
        embeddings_textual_evedences_train_random,true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, random_data_train, model,true_statements_embeddings, 'random',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(random_data_train):
        #     p = self.entityToNLrepresentation(p)
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_random.append(avg_embedding)
        embeddings_textual_evedences_train_range,true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, range_data_train, model,true_statements_embeddings, 'range',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(range_data_train):
        #     p = self.entityToNLrepresentation(p)
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train_range.append(avg_embedding)

        # ///////////////////////////////////////////////////////////////////////////////test
        # embeddings_textual_evedences_test_date = []
        # embeddings_textual_evedences_test_domain = []
        # embeddings_textual_evedences_test_domainrange = []
        # embeddings_textual_evedences_test_mix = []
        # embeddings_textual_evedences_test_property = []
        # embeddings_textual_evedences_test_random = []
        # embeddings_textual_evedences_test_range = []
        true_statements_embeddings_test = {}
        embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir,date_data_test,model, true_statements_embeddings_test, 'date_test',n)
        embeddings_textual_evedences_test_domain,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, domain_data_test, model,true_statements_embeddings_test, 'domain_test',n)
        embeddings_textual_evedences_test_domainrange, true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, domainrange_data_test, model, true_statements_embeddings_test, 'domainrange_test',n)
        embeddings_textual_evedences_test_mix,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, mix_data_test, model,true_statements_embeddings_test, 'mix_test',n)
        embeddings_textual_evedences_test_property,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir,property_data_test, model,true_statements_embeddings_test, 'property_test',n)
        embeddings_textual_evedences_test_random,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, random_data_test, model,true_statements_embeddings_test, 'random_test',n)
        embeddings_textual_evedences_test_range,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, range_data_test, model,true_statements_embeddings_test, 'range_test',n)
        # for idx, (s, p, o, c, p2,t1) in enumerate(date_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_date.append(avg_embedding)
        #
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(domain_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_domain.append(avg_embedding)
        #
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(domainrange_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_domainrange.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(mix_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     # Sentences are encoded by calling model.encode()
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_mix.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(property_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     # Sentences are encoded by calling model.encode()
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_property.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(random_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_random.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(range_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_range.append(avg_embedding)

        # /////////////////////////////////////////////////////////////////////////////// normal
        # embeddings_textual_evedences_train = self.getSentenceEmbeddings(self, data_train, model)
        # embeddings_textual_evedences_test = self.getSentenceEmbeddings(self, data_test, model)
        # for idx, (s, p, o, c, p2,t1) in enumerate(data_train):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train.append(avg_embedding)
        #
        #
        #
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     print(avg_embedding)
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #         print(avg_embedding)
        #         # exit(1)
        #
        #     embeddings_textual_evedences_test.append(avg_embedding)

        # /////////////////////////////////////////////////////////////////////////////// saving part
        # exit(1)


        # path = data_dir+ "data/train/date/"
        # X = np.array(embeddings_textual_evedences_train_date)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path + 'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/domain/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_domain, "train")

        # X = np.array(embeddings_textual_evedences_train_domain)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/domainrange/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_domainrange, "train")

        # X = np.array(embeddings_textual_evedences_train_domainrange)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/mix/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_mix, "train")

        # X = np.array(embeddings_textual_evedences_train_mix)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/property/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_property, "train")

        # X = np.array(embeddings_textual_evedences_train_property)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/random/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_random, "train")

        # X = np.array(embeddings_textual_evedences_train_random)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path + 'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/range/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_range, "train")

        # X = np.array(embeddings_textual_evedences_train_range)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)


        # /////

        # path = data_dir + "data/test/date/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_date, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_date)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/domain/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_domain, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_domain)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/domainrange/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_domainrange, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_domainrange)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path + 'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/mix/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_mix, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_mix)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/property/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_property, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_property)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/random/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_random, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_random)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/range/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_range, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_range)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)
        # /////////////////////////////////////////////////////////////////////////normal


        # X = np.array(embeddings_textual_evedences_train)
        # print(X.shape)
        # X=pd.DataFrame(X)
        # compression_opts = dict(method='zip',archive_name='trainSE.csv')
        # X.to_csv(data_dir+'trainSE.zip', index=False,compression=compression_opts)
        #
        # Y = np.array(embeddings_textual_evedences_test)
        # print(Y.shape)
        # Y=pd.DataFrame(Y)
        # compression_opts1 = dict(method='zip',archive_name='testSE.csv')
        # Y.to_csv(data_dir+'testSE.zip', index=False,compression=compression_opts1)






path_dataset_folder = './dataset/'
se = SentenceEmbeddings(data_dir=path_dataset_folder,multiclass=True)
