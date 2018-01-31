from collections import defaultdict
import numpy as np

def create_vocabulary(inputs, key_words=[]):
    """
    Create a vocabulary object from a sentence or a list of sentence

    :param inputs:
    :param vocab_size:
    :param key_words:
    :return:
    """

    vocab = Vocabulary()
    vocab.set_key_words(key_words)

    if isinstance(inputs[0], list):

        # inputs is a list of sentence
        for sentence in inputs:
            vocab.update_words(sentence)

    else:
        # inputs is a list of words
        vocab.update_words(inputs)

    vocab.sort()
    return vocab


class Vocabulary:

    def __init__(self, max_vocab_size=None):
        """
        init the vocabulary
        
        Vocabulary is a data structure to transform between word and its word index 

        :param vocab_size: size limitation of vocabulary
        """

        # word to id
        self.word_id = defaultdict()

        # id to word
        self.words = []

        # count distinct word in the dictionary
        self.vocab_size = 0

        # max vocabulary size
        self.max_vocab_size = max_vocab_size

        # count word frequencies
        self.word_freq = defaultdict(float)
        self.word_freq_sum = 0.0

        # stop word list
        # all words stored in this list should be excluded
        self.stop_words = []

        # key word list
        # all words stored in this list should be included in the dictionary despite their frequencies
        self.key_words = []

    def __str__(self):
        return '<Vocabulary: '+str(len(self.words))+' words>'

    def __repr__(self):
        return self.__str__()

    def set_stop_words(self, word_lst):
        """
        set up the stop words
        stop words will always be removed from sentences

        :param word_lst:
        :return:
        """
        self.stop_words = word_lst

    def set_key_words(self, word_lst):
        """
        set up the key words
        key words will never be removed from sentences despite their frequencies

        :param word_lst: a list containing key words
        :return:
        """
        self.key_words = word_lst

    def update_word(self, word):
        """
        Update the frequency for a word
        :param word: a single word
        :return:
        """

        if word not in self.word_id:
            self.words.append(word)
            self.word_id[word] = self.vocab_size
            self.vocab_size += 1

        self.word_freq[word] += 1.0
        self.word_freq_sum += 1.0

    def update_words(self, words):
        """
        update the frequency for words

        :param words: a list of word
        :return:
        """

        for word in words:
            self.update_word(word)

    def get_id(self, word):
        """
        get the id for a specific word

        :param word:
        :return: assigned word id or -1 if not found in the vocabulary boundary
        """

        if word in self.word_id:
            word_id = self.word_id[word]
            return word_id

        # return id for <UNK> if we have it
        if '<UNK>' in self.word_id:
            return self.word_id['<UNK>']
        else:
            return -1

    def get_ids(self, words):
        """
        get index for a word list
        :param words:
        :return:
        """

        ids = []
        for word in words:
            word_id = self.get_id(word)
            if word_id >= 0:
                ids.append(self.get_id(word))

        return ids

    def has_word(self, word):
        return word in self.word_id

    def get_word(self, word_id):
        if 0<= word_id < self.vocab_size:
            return self.words[word_id]
        else:
            return "<UNK>"

    def get_words(self, ids):
        """
        get original word from ids

        :param ids: word id list
        :return: a list of word
        """

        words = []
        for id in ids:
            words.append(self.get_word(id))

        return words

    def convert(self, x):
        """
        convert x into word or word id

        :param x: word or word id or its list or its list of list
        :return:
        """

        if len(x) == 0:
            return []

        result = []

        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
            if isinstance(x[0], int) or isinstance(x[0], np.int64) or isinstance(x[0], np.int32):
                result = self.get_words(x)
            elif isinstance(x[0], str):
                result = self.get_ids(x)
            else:
                #x is a list of list
                for lst in x:
                    if isinstance(lst[0], int) or isinstance(x[0], np.int64) or isinstance(x[0], np.int32):
                        result.append(self.get_words(lst))
                    else:
                        result.append(self.get_ids(lst))

        else:
            if isinstance(x, int) or isinstance(x[0], np.int64) or isinstance(x[0], np.int32):
                result = self.get_word(x)
            else:
                result = self.get_id(x)

        return result


    def sort(self):

        """
        rerank the whole dictionary based on there frequence
        however, stop word should always be excluded from the dictionary,
        and key words should always be included in the dictionary

        :return:
        """

        # update the frequency for words
        sort_word_freq = sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True)

        # unique word count
        word_index = 0

        # recreate word list and word id
        self.words = []
        self.word_id = defaultdict()

        # append the key words firstly
        for word in self.key_words:
            self.word_id[word] = word_index
            self.words.append(word)
            word_index += 1

        # remap the wordid
        for word_freq_pair in sort_word_freq:
            word = word_freq_pair[0]

            # skip the word if the word is in key words or stop words
            if word in self.word_id or word in self.stop_words:
                continue

            self.word_id[word] = word_index
            self.words.append(word)

            word_index += 1

        self.vocab_size = len(self.words)