
list_all_entities = []
with open('/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/pg_ranks/all_entities.txt' , "r") as f:
    for datapoint in f:
        final_str = datapoint.split('/')[-1].replace('>','')
        print(final_str)
        list_all_entities.append(final_str)

with open('/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/pg_ranks/all_entities_partsed.txt' , "w") as f:
    for item in list_all_entities:
        f.write("%s" % item)
