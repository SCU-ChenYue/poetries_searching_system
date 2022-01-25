# coding:utf-8
from gensim import corpora
from gensim.models import TfidfModel, LdaModel
from model_TFIDF_LDA import get_stopwords
from nltk.corpus import wordnet as wn
from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import csv
import gensim
from text2vec import Similarity
from gensim.models import KeyedVectors
from Levenshtein import *
import jieba.posseg as psg
from ltp import LTP
import functools
import synonyms
from util.utils import normalization, compare_right, compare_left, wash_list, wash_content, cut_sent

# 可接受词性 jieba
# flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an', 'ng', 'tg', 'vg', 'vd', 's', 'i', 'l']
flags = ['c', 'e', 'm', 'nh', 'o', 'p', 'r', 'u', 'wp', 'ws', 'x', 'd']  # 停用词性 LTP
# 标题,作者,原文句子,译文句子,赏析,原文分词,译文分词,赏析分词
corpus_path = "去重之后/总的去重合集/诗句、译文、赏析文件.csv"
embedding_path = "wordEmbedding/sgns.baidubaike.bigram-char"
tfidfPath = "checkpoint/checkpoint_100topics_100epoch/poetry_tfidf.model"
dicpath = "checkpoint/checkpoint_100topics_100epoch/poetry_dic.dict"
ldapath = "checkpoint/checkpoint_100topics_100epoch/poetry_lda.model"
allTagpath = "去重之后/分别提取的数据/tag.txt"
stop_path = "stopwords.txt"
stopwords = get_stopwords(stop_path)
tfidfModel = TfidfModel.load(tfidfPath)
ldaModel = LdaModel.load(ldapath)
dictionary = corpora.Dictionary.load(dicpath)
model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
words_list = model.index_to_key  # 词向量文件中的词列表
ltp = LTP(path="base")
analyze_minScore = 2
trans_minScore = 2


def get_all_tags():
    tag_list = []
    f = open(allTagpath, "r", encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        line = line.strip().replace("\n", "")
        if line != "":
            tag_list.append(line)
    tag_list = list(set(tag_list))
    return tag_list


tagList = get_all_tags()  # 所有的诗歌标签


def get_corpus(path):
    corpus = []
    poetry_list = []
    f = csv.reader(open(path, 'r', encoding='utf-8'))
    filesname = []
    for i in f:  # 标题,作者,原文句子,译文句子,赏析,原文分词,译文分词,赏析分词
        if len(i) == 0 or i[0] == '标题':
            continue
        tags = []
        c_tags, t_tags, a_tags = i[5].split("，"), i[6].split("，"), i[7].split("，")
        tags.extend(c_tags)
        tags.extend(t_tags)
        tags.extend(a_tags)
        corpus.append(tags)
        filesname.append(i[0])
        poetry_list.append(i)
    bm25Model = bm25.BM25(corpus)
    return bm25Model, poetry_list


def tfidf_normalization(qTFIDF):  # 对query的tifdf结果进行归一化
    scores = []
    tfidf_state = []
    for item in qTFIDF:
        scores.append(item[1])
    scores = normalization(scores)
    for i in range(len(qTFIDF)):
        tfidf_state.append((qTFIDF[i][0], scores[i]))
    return tfidf_state


# item：分数，标题，作者，朝代，原文，翻译，注释，赏析，标签
def LDA_sim(str1, str2):  # 判断所在诗歌内容的词语，是否是其关键主题词。
    seg1, hidden1 = ltp.seg([str1])
    seg2, hidden2 = ltp.seg([str2])
    vec_bow_1 = dictionary.doc2bow(seg1[0])
    vec_bow_2 = dictionary.doc2bow(seg2[0])
    vec_lda_1 = ldaModel[vec_bow_1]
    vec_lda_2 = ldaModel[vec_bow_2]
    sim = gensim.matutils.cossim(vec_lda_1, vec_lda_2)
    return sim


def cut_query(qlist, qSet_dict, k=5):  # 检索词组成的词语列表。应该在所有词替换完成后使用。
    # 先使用语料库中的tfidf进行打分
    corpus = [dictionary.doc2bow(qlist)]
    corpus_tfidf = list(tfidfModel[corpus])[0]
    corpus_tfidf.sort(key=functools.cmp_to_key(compare_right))
    token_state = []
    for item in corpus_tfidf:
        word = dictionary[item[0]]
        token_state.append((word, item[1]))
    token_state.sort(key=functools.cmp_to_key(compare_right))
    print("语料库的tf-idf打分情况：", token_state)
    token_state = token_state[:k]
    newQuery_list = [item[0] for item in token_state]
    keys = qSet_dict.keys()
    remove_keys = [key for key in keys if key not in newQuery_list]
    for key in remove_keys:
        qSet_dict.pop(key)
    return newQuery_list, qSet_dict


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


# 对搜索到的所有诗的结果进行排名。mode=1：若结果中包含了原始的、扩充前的query，则排在前面。mode=2：若结果包含了分别被扩充的多个query，则排在前面。
def get_result(query_context, qs, aidf, old_query, query_set_list):  # mode为0则不需要将包含了old_query的排在前面
    scores = BM25Model.get_scores(qs, aidf)  # 输入的文本内容必须是由关键词组成的列表
    paragraph_score = []
    for i in range(len(scores)):  # 分数，标题,作者,原文句子,译文句子,赏析
        paragraph_score.append((scores[i], poetryList[i][0], poetryList[i][1], poetryList[i][2], poetryList[i][3],
                                poetryList[i][4]))
    paragraph_score.sort(key=functools.cmp_to_key(compare_left))
    paragraph_score = [item for item in paragraph_score if item[0] > 0 and get_maxSenLength_fromPoetry(item) < 100]
    print("首先获取到的数据有：", len(paragraph_score))
    transList, anaList = [query_context], [query_context]
    for item in paragraph_score:
        transList.append(item[4])
        anaList.append(item[5])
    transEmbeddings = sentence_model.encode(transList)
    anaEmbeddings = sentence_model.encode(anaList)
    trans_similarity = cosine_similarity([transEmbeddings[0]], transEmbeddings[1:])
    ana_similarity = cosine_similarity([anaEmbeddings[0]], anaEmbeddings[1:])
    if len(trans_similarity[0]) != len(ana_similarity[0]):
        print("Wrong1！")
    if len(paragraph_score) != len(trans_similarity[0]):
        print("Wrong2！")

    poetry_scores = []  # 译文分数、标题、作者、原文句子、译文句子、赏析、赏析分数
    for i in range(len(paragraph_score)):
        poetry_scores.append([trans_similarity[0][i], paragraph_score[1], paragraph_score[2],
                              paragraph_score[3], paragraph_score[4], paragraph_score[5], ana_similarity[0][i]])
    poetry_scores.sort(key=functools.cmp_to_key(compare_right))  # 把赏析分数特别低的剔除
    poetry_scores = poetry_scores[:40]
    poetry_scores.sort(key=functools.cmp_to_key(compare_left))
    for item in poetry_scores:
        print("译文分数：", item[0], " 赏析分数：", item[6])
        print("标题：", item[1], " 作者：", item[2])
        print("原文：", item[3])
        print("译文：", item[4])
        print("赏析：", item[5])
        print("\n")
    return poetry_scores


def expand_query_setList(q, qExpansion, qset_Dict):  # q，qExpansion是根据q扩充的
    for qe in qExpansion:
        if qe != q and qe not in qset_Dict[q]:
            qset_Dict[q].append(qe)
    return qset_Dict


def same_stringPart(str1, str2):  # 判断两个字符是否有相同部分
    for i in str1:
        if i in str2:
            return True
    return False


def expanding_query_withDeleting(q_list, qset_dict, k):  # 依此使用wordNet,synonyms,word2vec进行近义词扩充
    print("进行扩充，每个词扩充", k, "个。")
    query_list = q_list.copy()
    for q in q_list:
        print("开始扩充", q)
        q_expansion = [q]  # 对于q，自己有一个扩充列表
        print("开始wordnet扩充。")
        wordnets = []
        wordnetSynset = wn.synsets(q, lang='cmn')
        for synset in wordnetSynset:
            wordnets.extend(synset.lemma_names('cmn'))
        wordnets = list(set(wordnets))
        if len(wordnets) >= 1:
            for i in range(len(wordnets)):
                wordnets[i] = wordnets[i].replace("+", "")
                if len(q_expansion) >= k + 1:  # 最多为k+1个，因为wordnet中的补充不删除
                    break
                if wordnets[i] in tagList and wordnets[i] not in query_list and wordnets[i] not in q_expansion:
                    if model.similarity(q, wordnets[i]) >= 0.5:
                        q_expansion.append(wordnets[i])
            if len(q_expansion) == k + 1:
                query_list.extend(q_expansion)
                print(q, " 的最终扩充结果为：", q_expansion)
                qset_dict = expand_query_setList(q, q_expansion, qset_dict)
                continue
        else:
            print("wordnet未获取到内容！")
        print("开始synonyms扩充。")
        synonyms_list = synonyms.nearby(q, 50)
        synonyms_words = synonyms_list[0]
        synonyms_scores = synonyms_list[1]
        if len(synonyms_words) >= 1:  # 至少得有内容
            for i in range(len(synonyms_words)):
                word, score = synonyms_words[i], synonyms_scores[i]
                if len(q_expansion) >= k + 2 or score < 0.5:
                    break
                if word not in query_list and word in tagList and word not in q_expansion:
                    q_expansion.append(word)
            if len(q_expansion) == 2 and k == 1:
                query_list.extend(q_expansion)
                print(q, "的最终扩充结果为：", q_expansion)
                qset_dict = expand_query_setList(q, q_expansion, qset_dict)
                continue
            elif len(q_expansion) >= 3:
                different_token = model.doesnt_match(q_expansion)
                if different_token != q and same_stringPart(different_token, q) is False:
                    print("删除：", different_token)
                    q_expansion.remove(different_token)
                elif different_token != q and same_stringPart(different_token, q) is True:
                    flag = 0
                    for token in reversed(q_expansion):
                        if token != q and same_stringPart(token, q) is False:
                            print("删除：", token)
                            q_expansion.remove(token)
                            flag = 1
                            break
                    if flag == 0:
                        print("删除：", different_token)
                        q_expansion.remove(different_token)
                else:  # doesn't match的刚好为q，则直接删除最后一个。
                    print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                if len(q_expansion) == k + 1:  # 若已经够了
                    query_list.extend(q_expansion)
                    print(q, "的最终扩充结果为：", q_expansion)
                    qset_dict = expand_query_setList(q, q_expansion, qset_dict)
                    continue
        else:
            print("synonyms未获取到内容！")
        print("开始word2vec扩充。")
        if q not in words_list:
            print("word2vec未获取到内容！")
            print(q, "的最终扩充结果为：", q_expansion)
            qset_dict = expand_query_setList(q, q_expansion, qset_dict)
            continue
        result_list = model.most_similar(q, topn=100)
        for item in result_list:
            word, score = item[0], item[1]
            if len(q_expansion) >= k + 2 or score < 0.5:
                break
            if word in tagList and word not in q_expansion and word not in query_list:
                q_expansion.append(word)
        if len(q_expansion) == 2:
            query_list.extend(q_expansion)
        elif len(q_expansion) == 1:
            print("没有扩充成功！")
        elif len(q_expansion) >= 3:
            if len(q_expansion) <= k + 1:
                query_list.extend(q_expansion)
            else:
                different_token = model.doesnt_match(q_expansion)
                if different_token != q and same_stringPart(different_token, q) is False:
                    print("删除：", different_token)
                    q_expansion.remove(different_token)
                elif different_token != q and same_stringPart(different_token, q) is True:
                    flag = 0
                    for token in reversed(q_expansion):
                        if token != q and same_stringPart(token, q) is False:
                            print("删除：", token)
                            q_expansion.remove(token)
                            if different_token in query_list:
                                query_list.remove(different_token)
                            flag = 1
                            break
                    if flag == 0:
                        print("删除：", different_token)
                        q_expansion.remove(different_token)
                else:  # doesn't match的刚好为q，则直接删除最后一个。
                    print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                query_list.extend(q_expansion)
        print(q, "的最终扩充结果为：", q_expansion)
        qset_dict = expand_query_setList(q, q_expansion, qset_dict)
    query_list = list(set(query_list))
    return query_list, qset_dict


def replace_with_word2vec(token):  # 使用word2vec替换token成新的token。
    re_token = ""
    re_score = 0
    results = model.most_similar(token, topn=100)
    for item in results:
        if item[0] in tagList:
            print("word2vec替换结果：", item[0])
            re_token = item[0]
            re_score = item[1]
            break
    if re_token == "":
        print("word2vec替换失败。")
    return re_token, re_score


def replace_synonyms(qlist, qSet_dict):  # 遍历分词后的word列表，并将不在数据集的词表中的词语换成数据集中的近义词
    queryList = []
    for token in qlist:
        if token not in tagList:
            # print("需要替换：", token)
            qSet_dict.pop(token)
            if token not in words_list:
                # print("LTP分词后无法被word2vec替换：", token)
                if len(token) > 2:  # 词长大于2时才考虑重新分词。
                    token_jieba = psg.cut(token)
                    tagFlag = 0
                    for word, flag in token_jieba:
                        if word in tagList:  # 如果刚好在tagList，则该词直接替换完成。
                            # print("JIEBA分词后，可以直接使用：", word)
                            queryList.append(word)
                            qSet_dict[word] = [word]
                            tagFlag = 1
                        if word not in tagList and word in words_list:  # 若不在tagList但是在word2vec中。
                            re_token, re_score = replace_with_word2vec(word)
                            # print("JIEBA分词后，可以用word2vec替换：", word, " 为：", re_token)
                            queryList.append(re_token)
                            qSet_dict[re_token] = [re_token]
                            tagFlag = 1
                    if tagFlag == 1:
                        continue
                    # 若没有被换掉，再看一次是否在word2vec中。
                    # print("JIEBA分词后不可以直接或间接替换。")

            # if count == 3 or (count == 2 and item[1] < 0.5):    # 多拓展1或2个
            #     break
            # if item[0] in tagList:
            #     print("替换结果：", item[0])
            #     queryList.append(item[0])
            #     qSet_dict[token].append(item[0])
            #     count += 1
            # break
            else:
                re_token, re_score = replace_with_word2vec(token)
                if re_score >= 0.5:
                    # print("直接使用word2vec将：", token, " 替换成：", re_token)
                    queryList.append(re_token)
                    qSet_dict[re_token] = [re_token]
                else:
                    if len(token) >= 2:
                        token = token[:len(token) - 1]
                        # print("切割")
                        if token in tagList:  # 如果刚好在tagList，则该词直接替换完成。
                            # print("替换为：", token)
                            queryList.append(token)
                            qSet_dict[token] = [token]
                        if token not in tagList and token in words_list:  # 若不在tagList但是在word2vec中。
                            ree_token, ree_score = replace_with_word2vec(token)
                            # print("用word2vec替换：", token, " 为：", ree_token)
                            queryList.append(ree_token)
                            qSet_dict[ree_token] = [ree_token]
        else:  # 不需要被替换
            queryList.append(token)
            qSet_dict[token] = [token]
    return queryList, qSet_dict


def process_query(context):
    seg, hidden = ltp.seg([context])
    seg = seg[0]
    pos = ltp.pos(hidden)[0]
    qs = []     # 保留进行了筛选、扩充后的query。
    for i in range(len(seg)):  # 对输入的query进行分词，并去除停用词。
        word = seg[i].strip()
        flag = pos[i].strip()
        if word not in stopwords and flag not in flags:
            qs.append(word)
        if word == '她':
            qs.append("妻子")
            qs.append("姑娘")
    qs = list(set(qs))
    query_set_dict = {}  # {q1:[q1,...], q2:[q2,...], q3:[q3,...]}
    for q in qs:
        query_set_dict[q] = [q]
    print("根据停用词和停用词性筛选后的结果：", qs)
    qs, query_set_dict = replace_synonyms(qs, query_set_dict)  # 替换不在词表中的词
    print("进行近义词替换后的结果：", qs)
    print(query_set_dict)
    if len(qs) > 3:
        print("裁剪关键词至3个。")
        qs, query_set_dict = cut_query(qs, query_set_dict, k=3)
        print(qs)
        print(query_set_dict)
    old_qs = qs.copy()  # 进行扩充之前的query
    if len(qs) <= 3:  # 若此时词语很少
        qs, query_set_dict = expanding_query_withDeleting(qs, query_set_dict, k=2)
    print("初步检索的结果：")
    print(query_set_dict)
    print(qs)
    if not qs:
        return
    # 进行搜索
    paragraph_score = get_result(context, qs, average_idf, old_qs, query_set_dict)


if __name__ == '__main__':
    BM25Model, poetryList = get_corpus(corpus_path)
    average_idf = sum(map(lambda k: float(BM25Model.idf[k]), BM25Model.idf.keys())) / len(BM25Model.idf.keys())
    print("请开始输入，输入quit结束。")
    query = input()
    while query != "quit":
        process_query(query)
        print("请继续输入，输入quit结束：")
        query = input()
