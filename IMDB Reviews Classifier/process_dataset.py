import glob
import string
from collections import Counter
import pickle
import random

def input_transformation():
    vocabulary = []
    # stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
    puncts = list(set(string.punctuation))

    all_positive = glob.glob('aclImdb_v1/aclImdb/test/neg/*')
    all_negative = glob.glob('aclImdb_v1/aclImdb/test/pos/*')
    
    step = 1000
    
    for i, f in enumerate(all_positive + all_negative):
        
        if i == step:
            print(i, 25000)
            step += 1000

        review = open(f, "r")
        words = []
        for line in review:
            line_clear = line.lower()    
            words_line = line_clear.split(' ')
            words += words_line

        for w in words:
            for p in puncts:
                w = w.replace(p, '')
            vocabulary.append(w)
    
    # vocabulary = [w for w in vocabulary if w not in stop_words]
    # vocabulary = Counter(vocabulary)
    # vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)

    # print(len(vocabulary))
    # pickle.dump(vocabulary, open('transformed_data/vocabulary', 'wb'))    


def reconstruct_dataset():
    vocabulary = pickle.load(open('transformed_data/vocabulary', 'rb'))
    puncts = list(set(string.punctuation))
    words_id = {}
    cc = 0
    for w in vocabulary[10:10000]:
        words_id[w[0]] = cc
        cc += 1
    
    print(len(words_id))

    all_positive = glob.glob('aclImdb_v1/aclImdb/train/neg/*')
    all_negative = glob.glob('aclImdb_v1/aclImdb/train/pos/*')


    train_data = []
    step = 1000
    for i, f in enumerate(all_positive):
        vector = [0] * len(words_id)
        label = 1

        if i == step:
            print(i, 12500)
            step += 1000

        review = open(f, "r")
        words = []
        for line in review:
            line_clear = line.lower()    
            words_line = line_clear.split(' ')
            words += words_line

        for w in words:
            for p in puncts:
                w = w.replace(p, '')
            try:
                w_id = words_id[w]
                vector[w_id] = 1
            except KeyError:
                pass
        
        train_data.append((vector, label))

    step = 1000
    for i, f in enumerate(all_negative):
        vector = [0] * len(words_id)
        label = 0

        if i == step:
            print(i, 12500)
            step += 1000

        review = open(f, "r")
        words = []
        for line in review:
            line_clear = line.lower()    
            words_line = line_clear.split(' ')
            words += words_line

        for w in words:
            for p in puncts:
                w = w.replace(p, '')
            try:
                w_id = words_id[w]
                vector[w_id] = 1
            except KeyError:
                pass
        
        train_data.append((vector, label))
    
    random.shuffle(train_data)
    X_train = []
    Y_train = []
    for pair in train_data:
        X_train.append(pair[0])
        Y_train.append(pair[1])


    
    print(len(train_data))
    print(len(X_train))
    print(len(Y_train))

    pickle.dump(X_train, open('transformed_data/X_train', 'wb'))
    pickle.dump(Y_train, open('transformed_data/Y_train', 'wb'))

if __name__ == "__main__":
    # input_transformation()
    reconstruct_dataset()

