## Marjoriikka Ylisiurua, University of Helsinki 23 March 2016

import re
from pprint import pprint
import csv

#openFileName = "D://tunteet.csv"
#openFileName = "D://julkkikset.csv"
#openFileName = "D://terveys.csv"
openFileName = "D://autot.csv"

#outcomeFileName = "D://tunteetLauseittain.csv"
#outcomeFileName = "D://julkkiksetLauseittain.csv"
#outcomeFileName = "D://terveysLauseittain.csv"
outcomeFileName = "D://autotLauseittain.csv"

commentData = []
parsedBody = []
splitCommentsData = []
foundComments = 0
foundSentences = 0

outComeCSV = ["ketjuNro","virkeNro","otsikko","virke","comments","subcomments"]
csvNames = ["ketjuNro","virkeNro","otsikko","virke","comments","subcomments"]

## main:
# for each file: open file, read it, parse it. Finally, save the outcome

with open(openFileName) as data_file:
    data = csv.DictReader(data_file, delimiter=";")
    pprint("Data read to memory...")

    for row in data:
        parsedBody = re.split("!+|\.+|\?+",row["content"])
        for i in range(0,len(parsedBody)):
            sentenceData = {"ketjuNro":row["number"],"virkeNro":str(i+1),"otsikko":row["topic"],"virke":parsedBody[i],"comments":row["comments"],"subcomments":row["subcomments"]}
            foundSentences = foundSentences+1
            splitCommentsData.append(sentenceData)
        foundComments = foundComments+1

pprint("Sentences found: "+str(foundSentences))
pprint("Comments found: "+str(foundComments))

with open(outcomeFileName,"w",newline='') as csvfile:
    dictWriter = csv.DictWriter(csvfile, fieldnames=csvNames,delimiter=";")
    dictWriter.writeheader()
    for row in splitCommentsData:
        dictWriter.writerow(row)

pprint("Done!")
