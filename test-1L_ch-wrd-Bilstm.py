#use normalize.py before calling this file.
#this file read a textfile and print its language using word and character features.

import sys
import warnings

import tensorflow
import pickle
from os import listdir
import xml.etree.ElementTree as ET
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax
import numpy as np
inFileName="1L_ch-wrd-BiLSTM" # for storing all files related to this model

def parseXML(xmlfile):
    #This function return text in comments tag of xml
    # create element tree object
    tree = ET.parse(xmlfile)
    # get root element
    root = tree.getroot()
    text=root[2].text.encode('utf8')
    # return news items list
    return text.decode("utf-8")

# load doc into memory
def load_doc(filename):
    # print("load_doc")
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def load_data_and_labels(foldername):
    sents, labels = [], []
    maxlen = 0
    linecnt = 0
    for fold in listdir(foldername):
        for file in listdir(foldername+"/"+fold):
            text=parseXML(foldername+"/"+fold+"/"+file)
            for line in re.split('\n|\?|।|!', text):  # .split('\n'):
                #line = line.lower()
                line = line.strip()  # remove blank spaces
                line = re.sub(' +', ' ', line)  # remove multiple spaces
                #line = line.translate(table)  # remove punctuations
                if len(line) < 1:
                    continue
                # store
                sents.append(line)
                labels.append(fold)
            # print(line)
    return sents, labels

def load_file_and_labels(foldername):
    postpara, labels = [], []  # hold a paragraph of facebook post
    for fold in listdir(foldername):
        for file in listdir(foldername+"/"+fold):
            text=load_doc(foldername+"/"+fold+"/"+file)
            # store
            postpara.append(text)
            labels.append(fold +"/"+file)
            # print(line)
    return postpara, labels

def load_model_and_Files():

    global encoder
    filename = inFileName + '-label_encoder.pkl'
    f=open(filename, 'rb')
    encoder= pickle.load(f)
    f.close()

    global char_tokenizer
    filename=inFileName+'-chartokenizer.pkl'
    f = open(filename, 'rb')
    char_tokenizer=pickle.load(f)
    f.close()

    global wrdtokenizer
    filename = inFileName + '-wrdtokenizer.pkl'
    f = open(filename, 'rb')
    wrdtokenizer = pickle.load(f)
    f.close()

    global model
    model=load_model(inFileName +'-best.hdf5')
    model._make_predict_function()
#    global graph
#    graph = tensorflow.get_default_graph()

    print("Model Loaded")

# map an integer to a word
def word_for_label_id(integer):
    return encoder.inverse_transform(integer)

max_wrd_len=30
max_sent_length=100
#get a char seq and convert into ids and pad them
def getchardata(d,flag):
    chardata=[]
    for i in range(len(d)):
        tuple=[]
        # add start token if generating prev word seq
        if flag==1:
            seq=char_tokenizer.texts_to_sequences(["~"])
            seq=pad_sequences(seq,max_wrd_len,padding='post')
            tuple.append(seq[0])
        j=0
        #convert every word of line at position i in data into seq of character and get their ids, pad them and put in tuple
        for wrd in d[i].split():
            if flag==1 and j==len(d[i])-1:  #if prev word seq, then skip last word
                break
            if flag==3 and j==0:  #if nxt word seq, skip first word
                continue
            seq=char_tokenizer.texts_to_sequences([list(wrd)])
            seq=pad_sequences(sequences=seq,maxlen=max_wrd_len,padding='post')
            tuple.append(seq[0])
            j+=1
        #----------- for loop ends here
        if flag==3:  # if nxt word seq generating then add end line token after padding
            seq=char_tokenizer.texts_to_sequences(["~"])
            seq=pad_sequences(seq,max_wrd_len,padding='post')
            tuple.append(seq[0])
        #check if length of tuple is equal to max length, if not then pad with zero
        if(len(tuple)<max_sent_length):
            while len(tuple)!= max_sent_length :
                tuple.append([0]*max_wrd_len)
        chardata.append(np.array(tuple))
    return np.array(chardata)

load_model_and_Files()
# load dataset and albels from xml files
foldername = 'testdata-normal' #need to read all files to get max length
postpara,labels = load_file_and_labels(foldername)
rsltfile = open("rslt-1lyr.csv", 'w')

for i in range(len(postpara)):
    print ("line " + str(i))
    text=postpara[i]
    test_descriptions=[]
    for line in re.split('\n|\?|।|!', text):  # .split('\n'):
        # line = line.lower()
        line = line.strip()  # remove blank spaces
        line = re.sub(' +', ' ', line)  # remove multiple spaces
        # line = line.translate(table)  # remove punctuations
        if len(line) < 1:
            continue
        # store
        #print(line)
        test_descriptions.append(line)
    testseq= wrdtokenizer.texts_to_sequences(test_descriptions)
    testseq = pad_sequences(testseq, maxlen=max_sent_length,padding='post')
    charseq=getchardata(test_descriptions,2)
    # load the model
    yhat = model.predict([testseq,charseq], verbose=0)
    # convert probability to integer
    yhat = argmax(yhat, axis=1).tolist()

    # map integer to word
    pred_class=max(yhat,key=yhat.count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        word = word_for_label_id(pred_class)
        print(labels[i] + "\t" + word + "\n")
        rsltfile.write(labels[i] + "\t" + word + "\n")

rsltfile.close()
