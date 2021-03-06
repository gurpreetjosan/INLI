# INLI
Indian Native Language Identification

These codes are result of Shared task on Indian Native Language Identification. Codes are implemented using Keras with tensorflow as backend.

Identifying native language i.e. first language of the person based on his/her writing in second language has inherent difficulty and depends on the succesfull identification of patterns in writing style which are influenced by the native language of writer. Succeeful native language identification has numerous applications like in forensic, personalised training and author profiling. INLI 2018 shared task focus on identifying the native language for users from their comments on various Facebook news posts. This can be treated as as a multiclass supervised classification task. Six Indian native languages need to be identified in this task i.e. Tamil, Hindi, Kannada, Malayalam, Bengali and Telugu.

# Data
Data was provided by organizers of FIRE 2018 as a shared task. You have to obtain your copy of data by directly contacting:

Anand Kumar M, Assistant Professor, Dept of IT, NITK-Surathkal
Soman K P , CEN, Amrita Vishwa Vidyapeetham, Coimbatore, India 

# System Description:

We have tried the Bidirectionl LSTM network for the current task. Two models have been employed for the task. First model capture the word level features where as second model captures both the word level and character level. For using these models we need to preprocess the input data.

# Preprocessing:
We process the input text sentence by sentence. Each post is split into sentences based on characters “.”, “!”, and “?”. As LSTM network expects fixed length input, length of every sentence is fixed to be 100 words. Shorter sentences are padded by zero and longer sentences are splitted into two snetences. In case of splitting, to keep the context, last four tokens of sentence are also replicated as first four tokens in second part of sentence. The class label of each sentence is kept same as the label of the post. Punctuation marks plays important role in identifying native langugae so they are not removed. All punctuation marks are included as seperate token. Further capitalization information is not kept and all tokens are converted into lower case. 

# Word embeddings:
Word embeddings are generated for all the words in input text. One can use pretrained word embeddings. As the input data is post from social media and people usually writes this text casually so pretrained word embeddings may not be able cover the vocabulary of input data. Thus we decided to use randomly intialised word vectors.

# Models:
We trained two models. Model 1 consider only word level features where as model 2 captures word as well as character level features. Here only model 2 has been discussed. We developed a joint feature vectore by combining the features of word and characters of that word. Further, to capture the features in forward direction as well as backward direction, Bi-directional LSTM has been decided to use. As every word is a sequence of characters, we use Bi directional LSTM network to extract the character level features of a word. 
![Model 2](1L_ch-wrd-BiLSTM-plot.png "Joint model for chracter and word level features")
Word length of every word is fixed to be 30 characters. Every word with number of characters shorter than 30 are padded with zeros and longer words are truncated. Feature vector of every character is kept of size 50. The sequence of embedded vector is passed through bidirectional LSTM network. The output of this network is passed through dense layer to obtain the character level features. Corresponding to each character, 250 features are produced by network. Thus a feature vector of total 7500 features has been produced representing word in terms of character feature. Every sentence is also passed through Bi directional LSTM network to extract the  features of a word. Sentence length of every sentence is fixed to be 100 words. Every senetence with number of words shorter than 100 are padded with zeros and longer sentences are break apart into two. Feature vector of every word is kept of size 100. The sequence of embedded vector is passed through bidirectional LSTM network. The output of this network is passed through dense layer to obtain the word level features. Corresponding to each word, 250 features are produced by network. The feature of word is joined with the feature vector of characters and this combined feature vector is passed again through Bidirectional LSTM network for joint learning. The output of Bidirectional LSTM is passed through dense layer which uses softmax function to produce the probability of six classes. 

# EXPERIMENTS & RESULTS
We used 90-10% split of the training data to validation split. Both the models were trained on CPU system with batch size of 32. Total 15 epochs were executed but in 8th epoch, validation acuracy surpass test accuracy signalling overfitting. Both model had a training accuracy of near 95%. Validation accuracy of first model is about 43% whereas for second model is 47%. We submitted two runs, run1 is the output of simple LSTM model where as run2 is the output of joint learned model on character and word features. Results shows that run1 is able to produce overall accuracy of 24.5% on test data whereas run2 is able to produce overall accuracy of 30.5%. Clearly run2 outperforms run1. i.e. joint model is able to capture more features than simple model.  
