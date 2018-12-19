from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras.utils as ku 
tokenizer = Tokenizer()



#Ngram

def ngram(data):
    
    # basic cleanup
    corpus = data.lower().split("\n") #Lower to reduce vocab size

    # tokenization
    tokenizer.fit_on_texts(corpus) #Creates a dictionary of all words and its index, {}'the' : 1, 'to': 2, ...} in tokenizer.word_index
    total_words = len(tokenizer.word_index) + 1 #Embedding layer expects input_dim to be vocabulary size + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    #The above basially creates a list of integers representing each sentence from n = 2 to n = len(sentence)

    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words


def generate_text_ngram(seed_text, next_words, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text


#Word2Vec
from gensim.models import Word2Vec
from tqdm import tqdm
from collections import Counter

def word2vec(data, window, min_count, sg): #CBOW model with negative sampling and 100 dimensional word vectors
    vocab = Counter()
    
    def text_to_wordlist(text):
        vocab.update(text)
        return text

    def process_comments(list_sentences):
        comments = []
        for text in tqdm(list_sentences):
            txt = text_to_wordlist(text)
            comments.append(txt)
        return comments
    
    tweets = process_comments(data)
    model = Word2Vec(tweets, size=100, window=window, min_count=min_count,  sg=sg)
    
    return model, vocab