import os
import math
import re
import numpy as np

TRAIN_PATH = "./train/"
VALID_PATH = "./valid/"
TEST_PATH = "./test/"


V1 = {}  # key: word1, value: count
V2 = {}  # key: (word1, word2), value: count
N = 0    # 所有单词（unigram）出现的次数总和
biN = 0  # 所有双词（bigram）出现的次数总和
La = 1  # Laplace Smoothing lambda
# punc = ['，', '。', '、', '：', '（', '）', '？', '！', '【', '】', '\n', '“', '”', '[', ']']
punc = ['\n', '“', '”', '[', ']']
# by_char = False
by_char = True
# method = "adding-one"
method = "good-turing"


def del_punc(word):
    ret = word
    for p in punc:
        ret = ret.replace(p, '')
    return ret


# 构建训练集对应的词库
if by_char:
    print("split mode: by char.")
else:
    print("split mode: by word.")
for file in os.listdir(TRAIN_PATH):
    with open(TRAIN_PATH + file, 'r', encoding='gbk') as f:
        wordlist = f.read().split('  ')
        wordlist = [del_punc(x.split('/')[0]) for x in wordlist]
        while '' in wordlist:
            wordlist.remove('')
        if by_char:
            tmp = []
            for x in wordlist:
                for xx in x:
                    tmp.append(xx)
            wordlist = tmp
        n = len(wordlist)
        N += n
        biN += n - 1
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
print("len(V1) =", len(V1), " len(V2) =", len(V2))
print("total number of word:", N)


def count(word, V):
    if word in V.keys():
        count = V[word]
    else:
        count = 0
    return count


def r_star(word, V, Nr):
    if word not in V.keys():  # 没有出现过的词
        return Nr[1]
    r = V[word]
    if r+1 not in Nr.keys():
        return r - 1  # 对于稀疏高频词（它的频率r，但不存在频率r+1的词）拟合结果
    else:
        return (r + 1) * Nr[r+1] / Nr[r]  # 否则使用Good Turing公式近似概率


uni_Nr = {}
bi_Nr = {}
for x in V1.keys():
    c = V1[x]
    if c in uni_Nr.keys():
        uni_Nr[c] += 1
    else:
        uni_Nr[c] = 1
for x in V2.keys():
    c = V2[x]
    if c in bi_Nr.keys():
        bi_Nr[c] += 1
    else:
        bi_Nr[c] = 1


# 估计句子的概率
def run(path, la, method):
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
                if by_char:
                    tmp = []
                    for x in wordlist:
                        for xx in x:
                            tmp.append(xx)
                    wordlist = tmp
                n = len(wordlist)
                if method == "adding-one":
                    _N = (N + B * La)
                    PPS1 = PPS2 = math.pow((count(wordlist[0], V1) + La) / _N, -1/n)
                    for i in range(1, n):
                        w1 = wordlist[i]
                        PPS1 *= math.pow((count(w1, V1) + La) / _N, -1/n)
                        if i < n - 1:
                            w2 = wordlist[i + 1]
                            PPS2 *= math.pow((count((w1, w2), V2) + La) / (count(w1, V1) + B * La), -1/n)
                    PPS1List.append(PPS1)
                    PPS2List.append(PPS2)
                elif method == "good-turing":
                    PPS1 = PPS2 = math.pow(r_star(wordlist[0], V1, uni_Nr) / N, -1/n)
                    for i in range(1, n):
                        w1 = wordlist[i]
                        PPS1 *= math.pow(r_star(w1, V1, uni_Nr) / N, -1/n)
                        if i < n - 1:
                            w2 = wordlist[i + 1]
                            PPS2 *= math.pow(r_star((w1, w2), V2, bi_Nr) / biN / (r_star(w1, V1, uni_Nr) / N), -1/n)
                    PPS1List.append(PPS1)
                    PPS2List.append(PPS2)

    ans1 = np.array(PPS1List)
    ans2 = np.array(PPS2List)
    print("PPS(Unigram):", np.mean(ans1))
    print("PPS(Bigram):", np.mean(ans2))  # unigram, bigram


la = 1
print("method: " + method)
print("--------  In valid dataset:  -------")
run(VALID_PATH, la, method)
print("--------  In test dataset:  -------")
run(TEST_PATH, la, method)

