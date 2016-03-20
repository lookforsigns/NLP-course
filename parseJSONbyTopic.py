## Marjoriikka Ylisiurua, University of Helsinki 20 March 2016

import json
import re
from pprint import pprint
import datetime
import codecs

### todo: 
## loop through every JSON file in folder
## now only a selected sample

searchToken = ["Avensis","Carina","Corolla","Yaris","Yleistä Toyotasta","Vähähiilihydraattiset dieetit","Yleistä laihdutuksesta","Laihduttajien tukiryhmä","Muut dieetit","Ikävä","Rakkaus","Ihastuminen","Mustasukkaisuus","Kuninkaalliset","Ulkomaiset julkkisjuorut","Missit"]
openFileName = ["D://dump58000-58099.json","D://dump02200-02299.json","D://dump02100-02199.json"]
##openFileName = ["D://dump02100-02199.json"]
##openFileName = ["D://sampleJSON 100 typerysta.json"]
outcomeFileName = "D://outcomeList.txt"
foundFileName = "D://FoundThreadList.json"

sampleComments = 0
listItemStartMsg = ""
listItemCommentMsg = ""
listThreadId = []
topic_IDs = []
foundInThread = 0
tokenFound = False

### sub findToken:
# find searchToken from some JSON element value
def findToken(searchToken, dicElement):
    for s in range(len(dicElement)):
        for l in range(len(searchToken)):
            if searchToken[l]==(dicElement[s]):
                return True
    return False

### main:
# for each file: open file, read it, analyse it. Finally, save the results on file

for item in range(len(openFileName)):
    with open(openFileName[item], encoding="utf-8") as data_file:
        data = json.load(data_file)
        pprint("Data read to memory...")

        ## check how many topics each thread has, append relevant topic titles to list

        for thread in data:
            for i in range(len(thread["topics"])):
                topic_IDs.append(thread["topics"][i]["title"])

        ## find topic titles from topic title list
        ## (search could also be made based on ID but they are not listed anywhere yet)
            if findToken(searchToken, topic_IDs):
                foundInThread = foundInThread+1 ## counter how many times titles are found
                tokenFound = True
            if tokenFound:
                listThreadId.append(thread)
                topic_IDs=[]
                tokenFound = False
                
## write resulting threads into file

##foundFile = open("D://FoundThreadList.json", "wb")
##foundFile = json.dumps(listThreadId)
##foundFile.close()

##pprint(listThreadId)
foundThreadJSON = json.dumps(listThreadId)

##pprint(foundThreadJSON)

foundFile = open(foundFileName, "w")
foundFile.write(foundThreadJSON)
#foundFile.write("[")
#for item in listThreadId:
#    json.dump(item, foundFile)
#    foundFile.write(',\n')
#foundFile.write("]")
foundFile.close()

pprint("Threads found: "+str(foundInThread)) 
pprint("File ready!")
