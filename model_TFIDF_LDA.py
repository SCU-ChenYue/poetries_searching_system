from gensim import corpora, models, similarities
from gensim.models import LdaModel
from util.utils import wash_content, get_stopwords
import re
import csv
from ltp import LTP
import jieba.posseg as psg
from pprint import pprint

# 参数定义
number_topics = 40
number_passes = 200
keep_number = 3000


def GenDictandCorpus(path):
    ltp = LTP(path="small")
    documents = []
    # flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an', 'ng', 'tg', 'vg', 'vd', 's', 'i', 'l']
    flags = ['c', 'e', 'm', 'nh', 'o', 'p', 'r', 'u', 'wp', 'ws', 'x', 'd', 'h', 'g', 'k', 'nd', 'ns', 'ni']
    stop_words = get_stopwords('LDAstopwords.txt')
    f = csv.reader(open(path, 'r', encoding='utf-8'))
    for i in f:
        if len(i) == 0:
            continue
        line = i[5] + i[7]
        document = []
        line = wash_content(line)
        line = re.sub(r"[\w\d]+", " ", line, flags=re.ASCII)
        seg, hidden = ltp.seg([line])
        seg = seg[0]
        pos = ltp.pos(hidden)[0]
        for i in range(len(pos)):
            word = seg[i].strip()
            flag = pos[i]
            if flag not in flags and word not in stop_words:
                document.append(word)
        documents.append(document)

        # for word, flag in psg.cut(line):
        #     word = word.strip()
        #     if flag in flags and word not in stop_words:
        #         document.append(word)
        # documents.append(document)
    print(len(documents))
    # 词典，输入的格式是：[[...],[...],[...]]
    dictionary = corpora.Dictionary(documents)
    # 压缩词向量，去掉出现的文章小于2的词，和在50%的文章都出现的词，整体长度限制在500
    # dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=500)
    dictionary.filter_extremes(no_below=2, no_above=0.3, keep_n=80000)
    # 词库，以(词，词频)方式存贮
    corpus = [dictionary.doc2bow(text) for text in documents]
    return dictionary, corpus


def LDA_and_TFIDF(path):
    dictionary, corpus = GenDictandCorpus(path)
    corpora.MmCorpus.serialize('checkpoint/checkpoint_11/train_corpuse.mm', corpus)
    dictionary.save("checkpoint/checkpoint_11/poetry_dic.dict")
    # 进行TF/IDF编码
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    tfidf.save("checkpoint/checkpoint_11/poetry_tfidf.model")
    # 主题个数
    num_topics = 100
    # 训练LDA模型，passes为迭代次数
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=100)
    lda.save("checkpoint/checkpoint_11/poetry_lda.model")
    # 打印主题
    for topic in lda.print_topics(num_words=num_topics, num_topics=num_topics):
        print(topic)


def test_LDA_topics(path, topics_num):
    lda = LdaModel.load(path)
    for topic in lda.print_topics(num_words=topics_num, num_topics=topics_num):  # 打印主题
        print(topic)


file_path = '去重之后/总的去重合集/格式化数据（赏析分段+赏析去重）.csv'
# LDA_and_TFIDF(file_path)
# model_path = 'checkpoint/checkpoint_1/poetry_lda.model'
# test_LDA_topics(model_path, 80)
