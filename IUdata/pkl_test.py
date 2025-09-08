import pickle
import os


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
file_path = os.path.join('D:\\fool\\Medical-Report-Generation-master\\IUdata', 'IUdata_vocab_0threshold.pkl')
# 读取.pkl文件
with open(file_path, 'rb') as f:
    vocab = pickle.load(f)

# 输出词汇表内容
for idx in range(len(vocab)):
    word = vocab.get_word_by_id(idx)
    print(f'ID: {idx}, Word: {word}')
