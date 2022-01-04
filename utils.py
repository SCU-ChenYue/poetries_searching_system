# coding:utf-8
import requests
from util.langconv import *
import jieba.posseg as psg
import numpy as np
from ltp import LTP
import re


# 请求html文件
def getHTMLText(url):
    try:
        kv = {'user-agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=kv)  # 防止查源user-agent
        r.raise_for_status()  # 确保200
        r.encoding = r.apparent_encoding  # 防止乱码
        demo = r.text
        return demo
    except:
        print("1爬取失败")


# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


# 自定义排序方式
def compare(a1, a2):
    if a1['score'] < a2['score']:
        return 1
    elif a1['score'] > a2['score']:
        return -1
    return 0


def compare_right(a1, a2):
    if a1[1] < a2[1]:
        return 1
    elif a1[1] > a2[1]:
        return -1
    return 0


def compare_left(a1, a2):
    if a1[0] < a2[0]:
        return 1
    elif a1[0] > a2[0]:
        return -1
    return 0


# 清洗中文文本的各项字符
def wash_content(token):
    bad_token = [" ", "\n", "\r", "\u3000", ".", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "…", "	",
                 "▲", "~"]
    for item in bad_token:
        token = token.replace(item, "")
    return token


def cut_sent(para):  # 分句
    para = re.sub('([。！；？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！；？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def cut_sent_mini(para):
    para = re.sub('([。！，：；”“"\….？\?])', "|", para)
    para = para.rstrip()
    return para.split("|")


def get_stopwords(path):
    stopwords = []
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line != "":
            stopwords.append(line)
    return stopwords


# 对文本分词（去除停用词和停用词性） jieba
def tokenizer(text):
    document = []
    word_flags = []
    text = wash_content(text)
    line = re.sub(r"[\w\d]+", " ", text, flags=re.ASCII)
    # 停用词性：n名词 nr人名 v动词 ns地名 nt机构团体名 d副词
    flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an', 'ng', 'tg', 'vg', 'vd', 's', 'i', 'l']
    # flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 'i', 'z']  # 构建tag时用的词性列表
    stop_words = get_stopwords('stopwords.txt')
    for word, flag in psg.cut(line):
        word = word.strip()
        if flag in flags and word not in stop_words:
            document.append(word)
            word_flags.append(flag)
    return document, word_flags


def tokenizer_LTP(text, ltp):
    document = []
    word_flags = []
    text = wash_content(text)
    line = re.sub(r"[\w\d]+", " ", text, flags=re.ASCII)
    flags = ['c', 'e', 'm', 'nh', 'o', 'p', 'r', 'u', 'wp', 'ws', 'x']
    stop_words = get_stopwords('stopwords.txt')
    seg, hidden = ltp.seg([line])
    seg = seg[0]
    pos = ltp.pos(hidden)[0]
    for i in range(len(pos)):
        word = seg[i].strip()
        flag = pos[i]
        if flag not in flags and word not in stop_words:
            document.append(word)
            word_flags.append(flag)
    return document, word_flags


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def wash_list(qlist):
    other_character = ["\n", "|", " ", "\u3000", "", "，", "。", "！", "？", "!", "?", ".", ",", '']
    query_list = []
    for item in qlist:
        if item not in other_character:
            if "|" in item:
                item = item.replace("|", "")
            query_list.append(item)
    return query_list


def wash_analyze(clist):
    analyze_list = []
    newcontent = ""
    for c in clist:  # 每一个段落
        paragraph = ""
        senList = cut_sent(c)
        for sen in senList:
            if "《" not in sen and "》" not in sen:
                paragraph += sen
                newcontent += sen
        analyze_list.append(paragraph)
    return analyze_list, newcontent    # 由每个段落组成的list [...,...


# test = ['“沧浪有钓叟，吾与尔同归”句，用形象表达观点，透露出韬光养晦才是真正的处世态度，愿意追随渔夫的思想和行迹，隐居江湖，不露才华。', '', '“处世忌太洁，至人贵藏晖”句，直接表达观点，明喻人生在世的处事方式，就应当如渔父那般不问是非、明哲保身的消极出世态度。', '']
# test = wash_list(test)
# print(test)
# for item in test:
#     print(item)
