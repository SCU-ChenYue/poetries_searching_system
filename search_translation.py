from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import functools
from util.utils import compare_left, wash_content
import csv
from ltp import LTP

ltp = LTP(path="base")
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
corpus_path = "去重之后/总的去重合集/诗句与译文文件.csv"
flags = ['m', 'o', 'p', 'q', 'r', 'wp', 'u', 'ws']
transList = []
poetry_items = []
# '标题', '作者', '原文句子', '译文句子', '原文分词', '译文分词'


def get_corpus(path):
    corpus = []
    poetry_list = []
    f = csv.reader(open(path, 'r', encoding='utf-8'))
    for i in f:
        if len(i) == 0:
            continue
        content_tags, trans_tags = i[4].split("，"), i[5].split("，")
        if len(trans_tags) != 0:
            corpus.append(trans_tags)
        else:
            corpus.append(content_tags)
        poetry_list.append(i)
    bm25Model = bm25.BM25(corpus)
    return bm25Model, poetry_list


def get_all_transSentence(path):
    f = csv.reader(open(path, "r", encoding="utf-8"))
    for i in f:
        if len(i) == 0:
            continue
        trans = wash_content(i[3])
        if trans == "":
            continue
        transList.append(trans)
        poetry_items.append([i[0], i[1], i[2], i[3]])
        if len(transList) != len(poetry_items):
            print("Wrong2!")


def get_SBert_scores(queryContext):
    test_embeddings = sentence_model.encode([queryContext])
    transList.append(queryContext)
    poetry_scores = []
    transEmbedding = sentence_model.encode(transList)
    trans_similarity = cosine_similarity([transEmbedding[-1]], transEmbedding[:-1])
    if len(trans_similarity[0]) != len(transList) - 1:
        print("Wrong!")
        return
    for i in range(len(trans_similarity[0])):
        trans_embedding = sentence_model.encode([poetry_items[i][3]])
        res = cosine_similarity([test_embeddings[0]], [trans_embedding[0]])
        poetry_scores.append([trans_similarity[0][i], res, poetry_items[i]])
        print(res)
    poetry_scores.sort(key=functools.cmp_to_key(compare_left))
    print("最低分是：", poetry_scores[-1])
    for item in poetry_scores[:20]:
        print(item)
    transList.remove(transList[-1])


def get_maxSenLength_fromPoetry(poetry_item):  # 返回最长的诗句长度
    contents = poetry_item[4]
    if "|" in contents:
        maxLen = 0
        clist = contents.split("|")
        for c in clist:
            if len(c) > maxLen:
                maxLen = len(c)
        return maxLen
    else:
        return len(contents)


# def process_query(context):
#     seg, hidden = ltp.seg([context])
#     pos = ltp.pos(hidden)[0]
#     seg = seg[0]
#     qs = []     # 保留进行了筛选、扩充后的query。
#     for i in range(len(seg)):  # 对输入的query进行分词，并去除停用词。
#         word = seg[i].strip()
#         flag = pos[i].strip()
#         if flag not in flags:
#             qs.append(word)
#     print(qs)
#     scores = BM25Model.get_scores(qs, average_idf)  # 输入的文本内容必须是由关键词组成的列表
#     paragraph_score = []
#     for i in range(len(scores)):  # '标题', '作者', '原文句子', '译文句子', '原文分词', '译文分词'
#         paragraph_score.append((scores[i], poetryList[i][0], poetryList[i][1], poetryList[i][2], poetryList[i][3]))
#     paragraph_score.sort(key=functools.cmp_to_key(compare_left))
#     paragraph_score = [item for item in paragraph_score if item[0] > 0 and len(item[2]) < 100]
#     for item in paragraph_score[:20]:
#         print("分数：", item[0], " 标题：", item[1], " 作者：", item[2])
#         print("原文：", item[3])
#         print("译文：", item[4])
#         print("\n")


if __name__ == '__main__':
    # BM25Model, poetryList = get_corpus(corpus_path)
    get_all_transSentence(corpus_path)
    # average_idf = sum(map(lambda k: float(BM25Model.idf[k]), BM25Model.idf.keys())) / len(BM25Model.idf.keys())
    print("请开始输入，输入quit结束。")
    query = input()
    while query != "quit":
        # process_query(query)
        get_SBert_scores(query)
        print("请继续输入，输入quit结束：")
        query = input()


