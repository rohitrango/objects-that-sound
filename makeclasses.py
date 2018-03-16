# coding: utf-8
import json

with open("ontology.json") as f:
    data = json.load(f)
    
d = dict([ (str(x['id']), str(x['name'])) for x in data])

with open("mappings.json", 'w+') as g:
    g.write(json.dumps(d))
    
