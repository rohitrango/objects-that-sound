# coding: utf-8
import json

# Ontology file contains all the metadata about 
# the classes, their human readable names, descriptions
with open("../metadata/ontology.json") as f:
    data = json.load(f)
    
# Just take the name and ID and store it in mappings.json
# This will contain ID to name mapping
d = dict([ (str(x['id']), str(x['name'])) for x in data])

with open("../metadata/mappings.json", 'w+') as g:
    g.write(json.dumps(d))
    
