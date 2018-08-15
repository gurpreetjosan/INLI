import re
from os import listdir
import xml.etree.ElementTree as ET
import os

sents, labels = [], []
maxlen = 0
linecnt = 0

def parseXML(xmlfile):
    #This function return text in comments tag of xml
    # create element tree object
    tree = ET.parse(xmlfile)
    # get root element
    root = tree.getroot()
    text=root[2].text.encode('utf8')
    # return news items list
    return text.decode("utf-8")

foldername='testdata'
outfoldername='testdata-normal'

if not os.path.exists(outfoldername):
    os.makedirs(outfoldername)

for fold in listdir(foldername):
    if not os.path.exists(outfoldername+"/"+fold):
        os.makedirs(outfoldername+"/"+fold)
    for file in listdir(foldername + "/" + fold):
        text = parseXML(foldername + "/" + fold + "/" + file)
        file=file.split('.')[0]+".txt"
        outfile = open(outfoldername + "/" + fold + "/" + file, 'w')
        for line in re.split('\n|\?|।|!', text):  # .split('\n'):
            # line = line.lower()
            line = line.strip()  # remove blank spaces
            line = re.sub(' +', ' ', line)  # remove multiple spaces
            # line = line.translate(table)  # remove punctuations
            if len(line) < 1:
                continue
            # store
            flag=True
            while(flag):
                if(len(line.split(" ")) > 100):
                    end=100
                else:
                    end=len(line.split(" "))
                    flag=False
                outfile.write(' '.join(line.split(" ")[:end]) + "\n")
                line=' '.join(line.split(" ")[end-5:])
            #outfile.write(line + "\n")
            # print(line)
            re.sub(r'(\\p{Punct})', r' \g<1> ', line)
        outfile.close()
