from urllib.request import urlopen
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import graph









# sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
# text = ""
# query = """PREFIX rdfs: http://www.w3.org/2000/01/rdf-schema#
#     PREFIX wd: http://www.wikidata.org/entity/
#     SELECT ?item
#     WHERE {
#     wd:"""+ids+""" rdfs:label ?item .
#     FILTER (langMatches( lang(?item), "EN" ) )
#     }
#     LIMIT 1 """
# sparql.setQuery(query)
# sparql.setReturnFormat(JSON)
# results = sparql.query().convert()
# text = results['results']['bindings'][0]['item']['value']
