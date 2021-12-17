class Data:
    def __init__(self, data_dir=None, subpath=None):

        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        if subpath==None:
            self.train_set = list((self.load_data(data_dir , data_type="train")))
            self.test_data = list((self.load_data(data_dir , data_type="test")))
        else:
            self.train_set = list((self.load_data(data_dir+"data/train/"+subpath, data_type="train")))
            self.test_data = list((self.load_data(data_dir+"data/test/"+subpath, data_type="test")))

        # factcheck predictions on train and test data
        if subpath==None:
            self.train_set_pred = list((self.load_data(data_dir , data_type="train_pred",pred=True)))
            self.test_data_pred = list((self.load_data(data_dir , data_type="test_pred",pred=True)))
        else:
            self.train_set_pred = list((self.load_data(data_dir+"data/train/"+subpath, data_type="train_pred",pred=True)))
            self.test_data_pred = list((self.load_data(data_dir+"data/test/"+subpath, data_type="test_pred",pred=True)))

        self.data = self.train_set + self.test_data
        self.entities = self.get_entities(self.data)
        self.save_all_resources(self.entities, data_dir,"data/combined/"+subpath, True)

        # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_data)))
        self.relations = self.get_relations(self.data)
        self.save_all_resources(self.relations, data_dir, "data/combined/" + subpath, False)
        self.idx_entities = dict()
        self.idx_relations = dict()

        # Generate integer mapping
        for i in self.entities:
            self.idx_entities[i] = len(self.idx_entities)
        for i in self.relations:
            self.idx_relations[i] = len(self.idx_relations)

        self.emb_entities = self.get_embeddings(self.idx_entities,'/home/umair/Documents/pythonProjects/HybridFactChecking/Embeddings/ConEx_dbpedia/','all_entities_embeddings')
        self.emb_relation = self.get_embeddings(self.idx_relations,'/home/umair/Documents/pythonProjects/HybridFactChecking/Embeddings/ConEx_dbpedia/','all_relations_embeddings')
        self.emb_sentences_train = self.get_sent_embeddings(data_dir+"data/train/"+subpath,'trainSE.csv')
        self.emb_sentences_test = self.get_sent_embeddings(data_dir+"data/test/"+subpath,'testSE.csv')

        self.idx_train_data = []
        for (s, p, o, label) in self.train_set:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_train_data.append([idx_s, idx_p, idx_o, label])

        self.idx_test_data = []
        for (s, p, o, label) in self.test_data:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_test_data.append([idx_s, idx_p, idx_o, label])


    @staticmethod
    def save_all_resources(list_all_entities, data_dir, sub_path, entities):
        if entities:
            with open(data_dir+sub_path+'all_entities.txt',"w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)
        else:
            with open(data_dir + sub_path + 'all_relations.txt', "w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)

    @staticmethod
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            assert label == 'True' or label == 'False'
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    # / home / umair / Documents / pythonProjects / HybridFactChecking / Embeddings / ConEx_dbpedia
    @staticmethod
    def get_embeddings(idxs,path,name):
        embeddings = dict()
        print("%s%s.txt" % (path,name))
        with open("%s%s.txt" % (path,name), "r") as f:
            for datapoint in f:
                data = datapoint.split('>,')
                if len(data)==1:
                    data = datapoint.split('>\",')
                if len(data) > 1:
                    data2 = data[0]+">",data[1].split(',')
                    test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    if test in idxs:
                        embeddings[test] = data2[1]
                    # else:
                        # print('Not in embeddings:',datapoint)
                # else:
                    # print('Not in embeddings:',datapoint)

        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)

        return embeddings_final.values()

    @staticmethod
    def get_sent_embeddings(path,name):
        embeddings = dict()
        print("%s%s" % (path,name))
        i =0
        with open("%s%s" % (path,name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    print(datapoint)
                else:
                    embeddings[i] = datapoint.split(',')
                    i = i+1

        return embeddings.values()



        # return embeddings.values()

