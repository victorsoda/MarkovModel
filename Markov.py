import os
import math
import re
import numpy as np

TRAIN_PATH = "./train/"
VALID_PATH = "./valid/"
TEST_PATH = "./test/"


V1 = {}  # key: word1, value: count
V2 = {}  # key: (word1, word2), value: count
N = 0   # len(V1)
cnt = 0
La = 1  # Laplace Smoothing lambda
# punc = ['，', '。', '、', '：', '（', '）', '？', '！', '【', '】', '\n', '“', '”', '[', ']']
punc = ['\n', '“', '”', '[', ']']


def del_punc(word):
    ret = word
    for p in punc:
        ret = ret.replace(p, '')
    return ret


# 构建训练集对应的词库
for file in os.listdir(TRAIN_PATH):
    with open(TRAIN_PATH + file, 'r', encoding='gbk') as f:
        wordlist = f.read().split('  ')
        wordlist = [del_punc(x.split('/')[0]) for x in wordlist]
        while '' in wordlist:
            wordlist.remove('')
        n = len(wordlist)
        N += n
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
        for i in range(n):
            w1 = wordlist[i]
            if w1 in V1.keys():
                V1[w1] += 1
            else:
                V1[w1] = 1
            if i+1 < n:
                w2 = wordlist[i+1]
                if (w1, w2) in V2.keys():
                    V2[(w1, w2)] += 1
                else:
                    V2[(w1, w2)] = 1
B = len(V1)


def count(word, V):
    if word in V.keys():
        count = V[word]
    else:
        count = 0
    return count


# 估计句子的概率
def run(path, la):
    La = la

    PPS1List = []   # 记录每一个句子的perplexity
    PPS2List = []

    for file in os.listdir(path):
        with open(path + file, 'r', encoding='gbk') as f:
            article = f.read()
            lines = article.split('\n')
            sentences = []
            for line in lines:
                if '。' in line:  # 加？！
                    s = re.split(r"([.。!！?？\s+]/w)", line)
                    s.append("")
                    s = ["".join(i) for i in zip(s[0::2], s[1::2])][:-1]
                    sentences.extend(s)
                elif len(line) > 3:  # 防止空行
                    sentences.append(line)

            for s in sentences:
                s = s.split('  ')
                wordlist = [del_punc(x.split('/')[0]) for x in s]
                while '' in wordlist:
                    wordlist.remove('')
                n = len(wordlist)
                _N = (N + B * La)
                PPS1 = PPS2 = math.pow((count(wordlist[0], V1) + La) / _N, -1/n)
                for i in range(1, n):
                    w1 = wordlist[i]
                    PPS1 *= math.pow((count(w1, V1) + La) / _N, -1/n)
                    if i < n - 1:
                        w2 = wordlist[i + 1]
                        PPS2 *= math.pow((count((w1, w2), V2) + La) / (count(w1, V1) + La), -1/n)
                PPS1List.append(PPS1)
                PPS2List.append(PPS2)

    ans1 = np.array(PPS1List)
    ans2 = np.array(PPS2List)
    print(np.mean(ans1))
    print(np.mean(ans2))


la = 1
run(VALID_PATH, la)
run(TEST_PATH, la)

