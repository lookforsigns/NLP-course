## Marjoriikka Ylisiurua 14 April 2016
## SNLP course project work

import csv
import json
from pprint import pprint

openFileName = "D://dataForLearning Labelled.csv"
outcomeFileName = "D://dataForLearning.json"
data = []
mydict = {}

# resulting file is constructed from selective parse and it has this title row
outcomeJSON = ("Label1","Label2","Label3","Label4","Label5","text","topic")

## main:
# for each file: open file, read it, parse it. Finally, save results on file

with open(openFileName,"r",encoding = "latin-1") as csvFile:
    reader = csv.DictReader(csvFile, outcomeJSON, delimiter=";") #sequence of dictionaries
    #reader = csv.DictReader(csvFile, delimiter=";") #sequence of dictionaries
    pprint("Data read to reader...")
 #   for row in reader:
 #       for i in range(0,len(outcomeJSON)):
 #           for row[i]:
 #               key = outcomeJSON[i]
 #               value = row[i]
 #               mydict[key] = value
 #       data.append(mydict) # list of dictionaries

    #data = {key: value for key,value in reader.items() if value is None}
    data = [row for row in reader]

 #   for row in data:
 #       if row["Certainty"]>0:
 #           row["Certainty"]=4
 #       else:
 #           row["Certainty"]       
 #   data["Label"]= data.pop("Certainty") 

with open(outcomeFileName,"w") as jsonFile:
    out = json.dumps(data)
    jsonFile.write(out)
