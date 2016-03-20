## Marjoriikka Ylisiurua, University of Helsinki 20 March 2016

import json
from pprint import pprint
import datetime
import csv
import codecs

##openFileName="D://dump58000-58099.json"
##openFileName ="D://sampleJSON 100 typerysta.json"
openFileName = "D://FoundThreadList.json"
outcomeFileName = "D://resultcsv.csv"
threadBasicData = []
foundComments = 0
foundErrors = 0

# resulting file is constructed from selective parse and it has this title row
outComeCSV = [["threadID","title","anonnick","body","created_at","parent_comment_ID","commentID","topic 1","topic 2","topic 3","topic 4","topic 5","topic 6","topic 7","topic 8"]]

## main:
# for each file: open file, read it, parse it. Finally, save the results on file

with open(openFileName, encoding="utf-8") as data_file:
    data = json.load(data_file)
    pprint("Data read to memory...")

    ## check how many topics each thread has, append those to separate list in backwards order (as there aren't as many topics for every subforum)
    ## pick up basic elements of thread and append with topiclist
    ## for whole row, add changing items: title, nick, timestamp, body, and id:s. for starting comment, "parent_comment_ID" is zero
    for thread in data:
        threadData = [thread["thread_id"],thread["title"]]
        threadData.append(thread["anonnick"])
        threadData.append(thread["body"])
        createdat = datetime.datetime.fromtimestamp(thread["created_at"]//1000)
        threadData.append(createdat)
        threadData.append(0)
        threadData.append(thread["thread_id"])
        for i in range(len(thread["topics"])):
            threadData.append(thread["topics"][len(thread["topics"])-i-1]["title"]) # start from last to first. this is not nice
        outComeCSV.append(threadData)
        foundComments=foundComments+1


        # what we are missing here is recursively go down to catch second, third etc. layer of comments
        for i in range(0,len(thread["comments"])):
            subcomment = thread["comments"][i]
            threadData = []
            threadData = [thread["thread_id"],thread["title"]]
            threadData.append(subcomment["anonnick"])
            threadData.append(subcomment["body"])
            createdat = datetime.datetime.fromtimestamp(subcomment["created_at"]//1000)
            threadData.append(createdat)
            threadData.append(thread["thread_id"])
            threadData.append(subcomment["comment_id"])
            for i in range(len(thread["topics"])):
                threadData.append(thread["topics"][len(thread["topics"])-i-1]["title"])
            ##print(threadData)
            outComeCSV.append(threadData)
            foundComments=foundComments+1
            ##print(outComeCSV)

## write resulting CSV-type element of lists into new csv file
# in case the text has non-unicode-8 characters, catch the error and pass
# (it has unicode-16 coding at least meaning we probably don't get actual JSON data but some dump)
outcomeFile = open(outcomeFileName,"w", newline='')
csvWriter = csv.writer(outcomeFile, delimiter=";")
for item in outComeCSV:
    try:
        csvWriter.writerow(item)
    except UnicodeEncodeError:
        foundErrors = foundErrors+1
outcomeFile.close()

pprint("Comments found: "+str(foundComments))
pprint("Errors found: "+str(foundErrors))
pprint("File ready!")
