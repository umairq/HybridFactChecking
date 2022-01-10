import sys
import csv
import requests
import tarfile
import json
import gzip
def main():
    print("start filtering")
    path = "/home/umair/Documents/pythonProjects/HybridFactChecking/Embeddings/TransE/"
    with open(path+"all_entity_embeddings_final.csv", 'w') as writer:
        print("")
    with open(path+"all_relation_embedding_final.csv", 'w') as writer:
        print("")

    entities = set()
    relations = set()

    entities_emb = dict()
    relations_emb = dict()

    # reading all training entiies
    with open("/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/complete_dataset/train.txt", 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 4:
                entities.add(datapoint[0])
                relations.add(datapoint[1])
                entities.add(datapoint[2])
            else:
                print(line)
                exit(1)

    # reading all testing entities
    with open("/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/complete_dataset/test.txt", 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 4:
                entities.add(datapoint[0])
                relations.add(datapoint[1])
                entities.add(datapoint[2])
            else:
                print(line)
                exit(1)
    for ttt in entities:
        print(ttt)
        parameters = {
            "entities" : [ttt.replace("<http://dbpedia.org","").replace(">","")],
            "indexname": "shallom_dbpedia_index"
        }
        headers = {
            'Content-type': 'application/j  son',
        }
        response = requests.get("http://unikge.cs.upb.de:5001/get-entity-embedding", data=json.dumps(parameters), headers=headers)

        if response.status_code==200:
            entities_emb.update(json.loads(response.content.decode('utf-8')))
            print(response.status_code)
            print(json.loads(response.content.decode('utf-8')))

    a_file = open(path+"all_entity_embeddings_final.csv", "w")
    writer = csv.writer(a_file)
    for key, value in entities_emb.items():
        writer.writerow([key, value])
    a_file.close()

    # with open(path+"", 'r') as f: ["/resource/Boeing_747_hull_losses"]
    #     for line in f:
    #         datapoint = line.split()
    #         if len(datapoint) == 4:
    #             entities.add(datapoint[0])
    #             relations.add(datapoint[1])
    #             entities.add(datapoint[2])
    #         else:
    #             print(line)
    #             exit(1)
    # with open(path +"", 'r') as f:
    #     for line in f:
    #         datapoint = line.split()
    #         if len(datapoint) == 4:
    #             entities.add(datapoint[0])
    #             relations.add(datapoint[1])
    #             entities.add(datapoint[2])
    #         else:
    #             print(line)
    #             exit(1)
    #
    #
    # print(len(entities))
    # result = []
    # count = 0
    # with gzip.open(file_emb, 'rb') as f:
    #     for line in f:
    #         print(str(line))
    #         if str(line).__contains__("<"):
    #             line =str(line).split("<")[1].replace("\\t","\t").replace("\\n","\n").replace("\\xc3\\xad","í")
    #             datapoint = line.split()
    #             if(datapoint[0].__contains__("resource")):
    #                 entity = datapoint[0].split("/resource/")
    #                 entity[-1] = entity[-1].replace("/", ".")
    #                 # print(entity)
    #                 # print(entity[-1][:-1])
    #                 if entities.__contains__(entity[-1][:-1]):
    #                     entity[-1] = entity[-1].replace(" ", "_").replace(",", "")
    #                     if datapoint[0].__contains__("dbpedia.org"):
    #                         if datapoint[0].__contains__("en.dbpedia.org") or datapoint[0].__contains__("//dbpedia.org") or datapoint[0].__contains__("//global.dbpedia.org"):
    #                             print(datapoint)
    #                             data = entity[-1][:-1]+" "+str(datapoint[1:]).replace("[",",").replace("]","").replace("'","").replace("\"","")
    #                             result.append(data)
    #                             count = count +1
    #                     else:
    #                         print(datapoint)
    #                         data = entity[-1][:-1] + " " + str(datapoint[1:]).replace("[", ",").replace("]", "").replace("'","").replace("\"","")
    #                         result.append(data)
    #         else:
    #             print("==================================>>>>>>>>>>>>>>>>>>>>>"+str(line))
    #         if len(result)>=100:
    #             with open(path+"entity_embedding_filter2.csv",'a') as f:
    #                 writer = csv.writer(f)
    #                 for l2 in result:
    #                     writer.writerow([l2])
    #             result.clear()
    #
    # print(count)
    # if len(result) >= 0:
    #     with open(path + "entity_embedding_filter2.csv", 'a') as f:
    #         writer = csv.writer(f)
    #         for l2 in result:
    #             writer.writerow([l2])
    #
    #     result.clear()
    # # relation embeddings
    #
    #
    # with open(file_emb_rel, 'r') as f:
    #     for line in f:
    #         datapoint = line.split()
    #         entity = datapoint[0].split("/")
    #         entity[-1] = entity[-1].replace(" ", "_").replace(",","")
    #         # print(entity)
    #         # print(entity[-1][:-1])
    #         if relations.__contains__(entity[-1][:-1]):
    #             if datapoint[0].__contains__("dbpedia.org"):
    #                 if datapoint[0].__contains__("en.dbpedia.org/ontology") or datapoint[0].__contains__("//dbpedia.org/ontology"):
    #                     if datapoint[1]=="rhs":
    #                         print(datapoint)
    #                         data = entity[-1][:-1]+" "+str(datapoint[5:]).replace("[",",").replace("]","").replace("'","")
    #                         result.append(data)
    #
    #             # else:
    #             #     print(datapoint)
    #             #     data = entity[-1][:-1] + " " + str(datapoint[5:]).replace("[", ",").replace("]", "").replace("'","")
    #             #     result.append(data)
    #
    #         if len(result)>=100:
    #             with open(path+"relation_embedding_filter2.csv",'a') as f:
    #                 writer = csv.writer(f)
    #                 for l2 in result:
    #                     writer.writerow([l2])
    #             result.clear()
    #
    #
    # if len(result) >= 0:
    #     with open(path + "relation_embedding_filter2.csv", 'a') as f:
    #         writer = csv.writer(f)
    #         for l2 in result:
    #             writer.writerow([l2])
    #
    #     result.clear()
    #
    #
    #
    #
    #

    exit(1)




















if __name__ == "__main__":
    main()
