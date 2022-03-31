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
import os
from gensim.models import KeyedVectors
from Levenshtein import *
import jieba.posseg as psg
from ltp import LTP
import functools
import synonyms
from util.utils import normalization, compare_right, compare_left, wash_list, wash_content, cut_sent, compare_left_small
from tqdm import tqdm

# 可接受词性 jieba
# flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an', 'ng', 'tg', 'vg', 'vd', 's', 'i', 'l']
flags = ['c', 'e', 'm', 'nh', 'o', 'p', 'r', 'u', 'wp', 'ws', 'x', 'd']  # 停用词性 LTP
# 标题,作者,原文句子,译文句子,赏析,原文分词,译文分词,赏析分词
corpus_path = "去重之后/总的去重合集/总数据.csv"
embedding_path = "wordEmbedding/sgns.baidubaike.bigram-char"
tfidfPath = "checkpoint/checkpoint_100topics_100epoch/poetry_tfidf.model"
dicpath = "checkpoint/checkpoint_100topics_100epoch/poetry_dic.dict"
ldapath = "checkpoint/checkpoint_100topics_100epoch/poetry_lda.model"
QAdata_path = "Experiments/experiment_data/知乎"
problem_path = "Experiments/未检索到结果的文件.txt"
log_path = "Experiments/results.txt"
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
analyze_minScore = 1.7
trans_minScore = 2
a = 1  # 译文和赏析各自所占分值的比重。为1时只看译文。
transList = []
analyzeList = []
poetry_items = []


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
    for i in f:
        if len(i) == 0 or i[0] == '标题':
            continue
        tags = []
        c_tags, t_tags, a_tags = i[5].split("，"), i[6].split("，"), i[7].split("，")
        tags.extend(c_tags)
        tags.extend(t_tags)
        tags.extend(a_tags)
        corpus.append(tags)
        i[0].strip()
        i[1].strip()
        i[2].strip()
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
    # print("语料库的tf-idf打分情况：", token_state)
    token_state = token_state[:k]
    newQuery_list = [item[0] for item in token_state]
    keys = qSet_dict.keys()
    remove_keys = [key for key in keys if key not in newQuery_list]
    for key in remove_keys:
        qSet_dict.pop(key)
    return newQuery_list, qSet_dict


def calculate_BM25_matchingScores(contents, translations, analyze_list):  # 计算每一段赏析和每句诗的BM25分数
    paragraphScore_Dict = {}  # {赏析的段落：[[分数、原文、译文], [分数、原文、译文]...]}
    for analyze in analyze_list:
        paragraphScore_Dict[analyze] = []
    sentence_analyze = []
    for i in range(len(contents)):  # 若原文和译文对齐，则依次遍历每一句话。
        sentence = contents[i]
        if len(translations) == 0:  # 有可能并没有传进来翻译
            trans = ""
        else:
            trans = translations[i]
        # print("用原文：", sentence, " 和译文：", trans, " 进行搜索。")
        corpus = []
        for paragraph in analyze_list:  # 遍历赏析的每一个文段
            seg, hidden = ltp.seg([paragraph])
            seg = seg[0]
            pos = ltp.pos(hidden)[0]
            wlist = []
            for k in range(len(seg)):
                word, flag = seg[k], pos[k]
                wlist.append(word)
            corpus.append(wlist)
        bm25Model = bm25.BM25(corpus)
        aidf = sum(map(lambda m: float(bm25Model.idf[m]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
        poetry_query, hidden = ltp.seg([sentence])  # 没有用trans
        scores = bm25Model.get_scores(poetry_query[0], aidf)  # 搜索得到该句话对应每个文段的分数
        paragraphs_score = []
        for j in range(len(scores)):
            paragraphs_score.append((scores[j], analyze_list[j]))
        paragraphs_score.sort(key=functools.cmp_to_key(compare_left))  # score分数的高低也要考虑，因为多个句子可以匹配到同样的赏析，但是匹配程度不同
        # print(paragraphs_score)
        for item in paragraphs_score:
            paragraphScore_Dict[item[1]].append([item[0], sentence, trans])
        if len(paragraphs_score) <= 1:
            paragraph = paragraphs_score[0][1]
        else:
            if paragraphs_score[0][0] <= paragraphs_score[1][0] * 3:  # 如果第一、二段的分数在三倍以内，则两段都要，
                paragraph = paragraphs_score[0][1] + paragraphs_score[1][1]
            else:
                paragraph = paragraphs_score[0][1]
        sentence_analyze.append([sentence, trans, paragraph])

    for key, value in paragraphScore_Dict.items():
        value.sort(key=functools.cmp_to_key(compare_left))
        paragraphScore_Dict[key] = value
    return sentence_analyze, paragraphScore_Dict


def sentence_fromPoetry(poetry_item, qs):  # 对于一首诗，利用原文、译文，从赏析中检索对应的文段，并返回该文段对应的句子。
    content, translation, analyze = poetry_item[4], poetry_item[5], poetry_item[7]
    content, translation, analyze = wash_content(content), wash_content(translation), wash_content(analyze)
    # 对赏析进行分段，并再分成短句后进行清洗。
    analyzes = analyze.split("|")
    # analyze_list, _ = wash_analyze(analyzes)
    analyze_list = analyzes
    contents, translations = content.split("|"), translation.split("|")
    analyzelist, contents, translations = wash_list(analyze_list), wash_list(contents), wash_list(translations)
    if len(contents) == 1:  # 如果默认分句的结果只有一句
        poetry_contents = cut_sent(content)  # 尝试手动分句
        trans_contents = cut_sent(translation)
        poetry_contents, trans_contents = wash_list(poetry_contents), wash_list(trans_contents)
        if len(poetry_contents) == 1:  # 若分完还是一句。
            query_list = []
            for qtt in qs:
                if qtt in analyze or qtt in content or qtt in translation:
                    query_list.append(qtt)
            return [0, content, translation, query_list, analyze]
        elif len(poetry_contents) != len(trans_contents) and len(poetry_contents) > 1:  # 长度大于1，但是不对齐。
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, [], analyzelist)
        else:  # 若手动分完后是对齐的了...
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, trans_contents, analyzelist)
    elif len(contents) == len(translations):  # 本来就对齐
        sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(contents, translations, analyzelist)
    else:  # 若不对齐
        poetry_contents = cut_sent(content)  # 先尝试手动分句
        trans_contents = cut_sent(translation)
        poetry_contents, trans_contents = wash_list(poetry_contents), wash_list(trans_contents)
        if len(poetry_contents) == 1:  # 若手动分句后只有一句
            query_list = []
            for qtt in qs:
                if qtt in analyze or qtt in content or qtt in translation:
                    query_list.append(qtt)
            return [0, content, translation, query_list, analyze]
        elif len(poetry_contents) != len(trans_contents):  # 若手动分句后有多句，但是不对齐，则只用原文去搜
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, [], analyzelist)
        else:  # 分句后对齐了
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, trans_contents, analyzelist)
    # scoreDict：{赏析的段落：[[分数、原文、译文], [分数、原文、译文]...]}，sentenceAnalyzes：[[sentence, trans, paragraph],...]
    # 首先通过是否出现在原文、译文的方式找关键句。

    maxCount = 0
    maxSentence = []  # [score, content, translation, [q1, q2, q3], analyze]
    for key, value in scoreDict.items():  # 找出包含最多keyword的赏析对应的最高和第二高句子。
        query_list = []
        count = 0
        for qm in qs:
            if qm in key:
                count += 1
                query_list.append(qm)
                # print(q)
        if count > maxCount:
            maxCount = count
            # 新的扩充算法（同首诗的某个赏析，对应的诗句）
            maxSentence = [value[0][0], value[0][1], value[0][2], query_list, key]
            if len(value) >= 2 and len(value[0][1]) >= 7 and (
                    value[0][0] <= value[1][0] + 0.5 or value[0][0] * 0.9 <= value[1][0]):
                maxSentence = [(value[0][0] + value[1][0]) / 2, value[0][1] + value[1][1], value[0][2] + value[1][2],
                               query_list, key]
            elif len(value[0][1]) < 7:  # 太短了也自动扩充
                if value[0][1][-1] != "。" or value[0][1][-1] != "，" or value[0][1][-1] != "！" or value[0][1][-1] != "？":
                    maxSentence = [
                        (value[0][0] + value[1][0]) / 2, value[0][1] + "。" + value[1][1], value[0][2] + value[1][2],
                        query_list, key]
                else:
                    maxSentence = [
                        (value[0][0] + value[1][0]) / 2, value[0][1] + value[1][1], value[0][2] + value[1][2],
                        query_list, key]
    if len(maxSentence) == 0:
        return maxSentence
    if float(maxSentence[0]) < 3:  # 可以考虑判断使用整诗，只要不太长（或整体分数还不错）。
        content, translation = content.replace("|", ""), translation.replace("|", "")
        analyze = analyze.replace("|", "")
        if len(content) <= 66:
            return [0, content, translation, maxSentence[3], analyze]
        else:
            return []
    flag = 0
    if len(contents) == len(translations):
        flag = 1
    if len(maxSentence[1]) <= 8:
        if len(contents) == 2:
            maxSentence[1] = content.replace("|", "")
        if len(contents) >= 3:
            for i in range(len(contents)):
                sen = contents[i]
                if distance(sen, maxSentence[1]) <= 1:
                    if i == 0:
                        maxSentence[1] += contents[i + 1]
                        if flag == 1:
                            maxSentence[2] += translations[i + 1]
                    elif i == len(contents) - 1:
                        maxSentence[1] = contents[i - 1] + maxSentence[1]
                        if flag == 1:
                            maxSentence[2] = translations[i - 1] + maxSentence[2]
                    else:
                        if len(contents[i - 1]) <= len(contents[i + 1]):
                            maxSentence[1] = contents[i - 1] + maxSentence[1]
                            if flag == 1:
                                maxSentence[2] = translations[i - 1] + maxSentence[2]
                        else:
                            maxSentence[1] = maxSentence[1] + contents[i + 1]
                            if flag == 1:
                                maxSentence[2] = maxSentence[2] + translations[i + 1]
                    break
    return maxSentence  # 函数返回的是包含了最多检索词的赏析所在的诗句


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
def get_result(query_context, qs, aidf, query_set_list):  # mode为0则不需要将包含了old_query的排在前面
    scores = BM25Model.get_scores(qs, aidf)  # 输入的文本内容必须是由关键词组成的列表
    paragraph_score = []
    for i in range(len(scores)):  # 分数,标题,作者,原文句子,译文句子,赏析
        paragraph_score.append((scores[i], poetryList[i][0], poetryList[i][1], poetryList[i][2], poetryList[i][3],
                                poetryList[i][4]))
    # paragraph_score = [item for item in paragraph_score if item[0] > 0 and get_maxSenLength_fromPoetry(item) < 100]
    paragraph_score.sort(key=functools.cmp_to_key(compare_left))
    paragraph_score = [item for item in paragraph_score if item[0] > 0 and get_maxSenLength_fromPoetry(item) < 100]
    # print("首先获取到的数据有：", len(paragraph_score))
    # return paragraph_score

    # paragraph_score = paragraph_score[:25]
    # query_set_list: {q1:[q1,...], q2:[q2,...], q3:[q3,...]}，根据出现在用户输入中的词的数量
    paragraph_score_count = []  # 每一个item：[3, [分数,标题,...], [key1, key2,...]]
    keys = query_set_list.keys()
    for item in paragraph_score:
        count = 0
        poetry_keyword = []
        for key in keys:
            q_set_list = query_set_list[key]
            flag = 0
            for qq in q_set_list:
                if qq in item[3] or qq in item[4] or qq in item[5]:
                    if flag == 0:
                        count += 1  # 统计同时是q1,q2,q3的列表中的词语
                        flag = 1
                    poetry_keyword.append(qq)
        paragraph_score_count.append([count, item, poetry_keyword])
    paragraph_score_count.sort(key=functools.cmp_to_key(compare_left))

    # 假如用户输入了3、4个关键词，则至少得到的结果要含有两个。
    # maxCount = paragraph_score_count[0][0]
    maxCount = len(keys)
    temp = paragraph_score_count
    if (maxCount == 4 or maxCount == 3) and len(paragraph_score_count) > 10:
        minCount = 2
        paragraph_score_count = [item for item in paragraph_score_count if item[0] >= minCount]
    # print("第一次筛选过后有：", len(paragraph_score_count))
    if len(paragraph_score_count) <= 20:
        paragraph_score_count = temp
    query_trans_Sentences, query_analyze_Sentences = [query_context], [query_context]
    for item in paragraph_score_count:
        query_trans_Sentences.append(item[1][4])
        query_analyze_Sentences.append(item[1][5])

    query_trans_embedding = sentence_model.encode(query_trans_Sentences)
    query_analyze_embedding = sentence_model.encode(query_analyze_Sentences)
    query_trans_similarity = cosine_similarity([query_trans_embedding[0]], query_trans_embedding[1:])
    query_analyze_similarity = cosine_similarity([query_analyze_embedding[0]], query_analyze_embedding[1:])

    # if len(query_trans_similarity[0]) != len(query_analyze_similarity[0]):
    #     print("Wrong1!")
    # if len(paragraph_score_count) != len(query_trans_similarity[0]):
    #     print("Wrong2!")

    SBert_scores = []  # 每一个item是：[analyze_score, trans_score, [score, content, translation, [q1, q2, q3], analyze]]
    maxQueryAnalyzeScore, maxQueryTransScore = 0, 0
    for i in range(len(query_trans_similarity[0])):
        if query_analyze_similarity[0][i] > maxQueryAnalyzeScore:
            maxQueryAnalyzeScore = query_analyze_similarity[0][i]
        if query_trans_similarity[0][i] > maxQueryTransScore:
            maxQueryTransScore = query_trans_similarity[0][i]
        SBert_scores.append([query_trans_similarity[0][i], query_analyze_similarity[0][i], paragraph_score_count[i]])
    # 先剔除一部分赏析分数低的，这一步作为初筛选。
    # print("赏析最高分为：", maxQueryAnalyzeScore)
    # print("译文最高分为：", maxQueryTransScore)
    SBert_scores = [item for item in SBert_scores if item[1] * analyze_minScore >= maxQueryAnalyzeScore]
    SBert_scores.sort(key=functools.cmp_to_key(compare_left))
    if len(SBert_scores) == 0:
        return SBert_scores
    # if SBert_scores[0][0] < 0.5:
    #     return []
    # print("最后还剩：", len(SBert_scores), "条数据。")
    # for item in SBert_scores[:10]:
    #     print("SBert对译文的分数：", item[0])
    #     print("SBert对赏析的分数：", item[1])
    #     print("文段的BM25分数：", item[2][1][0])
    #     print("文段的关键词：", item[2][2])
    #     print("原文：", item[2][1][3])
    #     print("译文：", item[2][1][4])
    #     print("赏析：", item[2][1][5])
    #     print("\n")
    return SBert_scores


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
    # print("进行扩充，每个词扩充", k, "个。")
    query_list = q_list.copy()
    for q in q_list:
        # print("开始扩充", q)
        q_expansion = [q]  # 对于q，自己有一个扩充列表
        # print("开始wordnet扩充。")
        wordnets = []
        try:
            wordnetSynset = wn.synsets(q, lang='cmn')
        except:
            continue
        for synset in wordnetSynset:
            wordnets.extend(synset.lemma_names('cmn'))
        wordnets = list(set(wordnets))
        if len(wordnets) >= 1:
            for i in range(len(wordnets)):
                wordnets[i] = wordnets[i].replace("+", "")
                if len(q_expansion) >= k + 1:  # 最多为k+1个，因为wordnet中的补充不删除
                    break
                if wordnets[i] in tagList and wordnets[i] not in query_list and wordnets[i] not in q_expansion and \
                        wordnets[i] in words_list:
                    if model.similarity(q, wordnets[i]) >= 0.5:
                        q_expansion.append(wordnets[i])
            if len(q_expansion) == k + 1:
                query_list.extend(q_expansion)
                # print(q, " 的最终扩充结果为：", q_expansion)
                qset_dict = expand_query_setList(q, q_expansion, qset_dict)
                continue
        # else:
        #     print("wordnet未获取到内容！")
        # print("开始synonyms扩充。")
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
                # print(q, "的最终扩充结果为：", q_expansion)
                qset_dict = expand_query_setList(q, q_expansion, qset_dict)
                continue
            elif len(q_expansion) >= 3:
                different_token = model.doesnt_match(q_expansion)
                if different_token != q and same_stringPart(different_token, q) is False:
                    # print("删除：", different_token)
                    q_expansion.remove(different_token)
                elif different_token != q and same_stringPart(different_token, q) is True:
                    flag = 0
                    for token in reversed(q_expansion):
                        if token != q and same_stringPart(token, q) is False:
                            # print("删除：", token)
                            q_expansion.remove(token)
                            flag = 1
                            break
                    if flag == 0:
                        # print("删除：", different_token)
                        q_expansion.remove(different_token)
                else:  # doesn't match的刚好为q，则直接删除最后一个。
                    # print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                if len(q_expansion) == k + 1:  # 若已经够了
                    query_list.extend(q_expansion)
                    # print(q, "的最终扩充结果为：", q_expansion)
                    qset_dict = expand_query_setList(q, q_expansion, qset_dict)
                    continue
        # else:
        #     print("synonyms未获取到内容！")
        # print("开始word2vec扩充。")
        if q not in words_list:
            # print("word2vec未获取到内容！")
            # print(q, "的最终扩充结果为：", q_expansion)
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
        # elif len(q_expansion) == 1:
        #     print("没有扩充成功！")
        elif len(q_expansion) >= 3:
            if len(q_expansion) <= k + 1:
                query_list.extend(q_expansion)
            else:
                different_token = model.doesnt_match(q_expansion)
                if different_token != q and same_stringPart(different_token, q) is False:
                    # print("删除：", different_token)
                    q_expansion.remove(different_token)
                elif different_token != q and same_stringPart(different_token, q) is True:
                    flag = 0
                    for token in reversed(q_expansion):
                        if token != q and same_stringPart(token, q) is False:
                            # print("删除：", token)
                            q_expansion.remove(token)
                            if different_token in query_list:
                                query_list.remove(different_token)
                            flag = 1
                            break
                    if flag == 0:
                        # print("删除：", different_token)
                        q_expansion.remove(different_token)
                else:  # doesn't match的刚好为q，则直接删除最后一个。
                    # print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                query_list.extend(q_expansion)
        # print(q, "的最终扩充结果为：", q_expansion)
        qset_dict = expand_query_setList(q, q_expansion, qset_dict)
    query_list = list(set(query_list))
    return query_list, qset_dict


def replace_with_word2vec(token):  # 使用word2vec替换token成新的token。
    re_token = ""
    re_score = 0
    results = model.most_similar(token, topn=100)
    for item in results:
        if item[0] in tagList:
            # print("word2vec替换结果：", item[0])
            re_token = item[0]
            re_score = item[1]
            break
    # if re_token == "":
    #     print("word2vec替换失败。")
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


def get_all_transSentence(path):
    f = csv.reader(open(path, "r", encoding="utf-8"))
    for i in f:
        if len(i) == 0 or i[0] == '标题':
            continue
        trans = wash_content(i[3])
        analyzes = wash_content(i[4])
        if trans == "" or analyzes == "":
            continue
        transList.append(trans)
        analyzeList.append(analyzes)
        poetry_items.append([i[0], i[1], i[2], i[3], i[4]])
        # if len(transList) != len(poetry_items):
        #     print("Wrong2!")
        # if len(transList) != len(analyzeList):
        #     print("Wrong3!")


def get_SBert_scores(queryContext, k):
    transList.append(queryContext)
    analyzeList.append(queryContext)
    poetry_scores = []
    transEmbedding = sentence_model.encode(transList)
    anaEmbedding = sentence_model.encode(analyzeList)
    trans_similarity = cosine_similarity([transEmbedding[-1]], transEmbedding[:-1])
    ana_similarity = cosine_similarity([anaEmbedding[-1]], anaEmbedding[:-1])
    if len(trans_similarity[0]) != len(transList) - 1 or len(trans_similarity[0]) != len(ana_similarity[0]):
        # print("Wrong!")
        return
    for i in range(len(trans_similarity[0])):
        poetry_scores.append([trans_similarity[0][i], poetry_items[i], ana_similarity[0][i]])
    poetry_scores.sort(key=functools.cmp_to_key(compare_left))  # 先按照译文的分数进行排名。
    rank_poetry_scores = []  # 译文排名、译文分数、诗歌信息[...]、赏析分数、赏析排名
    for index in range(len(poetry_scores)):
        item = poetry_scores[index]
        rank_poetry_scores.append([index + 1, item[0], item[1], item[2]])
    rank_poetry_scores.sort(key=functools.cmp_to_key(compare_right))
    for index in range(len(rank_poetry_scores)):
        rank_poetry_scores[index].append(index + 1)

    for i in range(len(rank_poetry_scores)):
        average_rank = (rank_poetry_scores[i][0] * a + rank_poetry_scores[i][4] * (1 - a))
        rank_poetry_scores[i] = [average_rank, rank_poetry_scores[i][1], rank_poetry_scores[i][3],
                                 rank_poetry_scores[i][2]]

    rank_poetry_scores.sort(key=functools.cmp_to_key(compare_left_small))
    # for item in rank_poetry_scores[:k]:  # [平均排名、译文分数、赏析分数、[标题、作者、原文、译文、赏析]]
    #     print("平均排名：", item[0])
    #     print("译文分数：", item[1], " 赏析分数：", item[2])
    #     print("标题：", item[3][0], " 作者：", item[3][1])
    #     print("原文：", item[3][2])
    #     print("译文：", item[3][3])
    #     print("赏析：", item[3][4])
    #     print("\n")

    transList.remove(transList[-1])
    analyzeList.remove(analyzeList[-1])


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


def clean_query(context):
    clean_tags = ['表达', '最', '形容', '描写', '表现', '通过', '写', '最', '体现', '抒发']
    for tag in clean_tags:
        if tag in context:
            context = context.replace(tag, "")
    return context


def process_query(context):
    # context = clean_query(context)
    context = context.replace(".csv", "")
    seg, hidden = ltp.seg([context])
    seg = seg[0]
    pos = ltp.pos(hidden)[0]
    qs = []  # 保留进行了筛选、扩充后的query。
    for i in range(len(seg)):  # 对输入的query进行分词，并去除停用词。
        word = seg[i].strip()
        flag = pos[i].strip()
        qs.append(word)
        # if word not in stopwords and flag not in flags:
        #     qs.append(word)
        # if word == '她':
        #     qs.append("妻子")
        #     qs.append("姑娘")
    qs = list(set(qs))
    query_set_dict = {}  # {q1:[q1,...], q2:[q2,...], q3:[q3,...]}
    for q in qs:
        query_set_dict[q] = [q]
    # print("根据停用词和停用词性筛选后的结果：", qs)
    qs, query_set_dict = replace_synonyms(qs, query_set_dict)  # 替换不在词表中的词
    # print("进行近义词替换后的结果：", qs)
    # print(query_set_dict)
    if len(qs) > 3:
        # print("裁剪关键词至3个。")
        qs, query_set_dict = cut_query(qs, query_set_dict, k=3)
        # print(qs)
        # print(query_set_dict)
    if len(qs) <= 3:  # 若此时词语很少
        qs, query_set_dict = expanding_query_withDeleting(qs, query_set_dict, k=2)
    # print("初步检索的结果：")
    # print(query_set_dict)
    # print(qs)
    paragraph_score = []
    if len(qs) != 0:
        # 进行搜索
        context = clean_query(context)
        paragraph_score = get_result(context, qs, average_idf, query_set_dict)
    return paragraph_score
    # if len(paragraph_score) == 0:
    #     get_SBert_scores(context, k=20)
    # if paragraph_score[0][0] < 0.5:
    #     get_SBert_scores(context, k=1)


def test(QAdata, K):
    fw = open(problem_path, "w", encoding="utf-8")
    flog = open(log_path, "w", encoding="utf-8")
    total_data = 0
    total_p_data = 0
    total_right = 0
    temp_count = 0
    total_pre = 0
    for root, dirs, files in os.walk(QAdata):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        print(len(files))
        for f in tqdm(files):
            # fwashed = csv.writer(open(washed_path + "/" + f, "w+", encoding="utf-8"))
            appear_number = 0
            content_list = []
            filename = f
            filepath = os.path.join(root, f)
            fr = csv.reader(open(filepath, "r", encoding="utf-8"))
            for i in fr:
                if len(i) == 0:
                    continue
                i[0].strip()
                i[1].strip()
                i[2].strip()
                content_list.append([i[0], i[1], i[2]])  # 标题，作者，原文句子
                total_data += 1
            if len(content_list) == 0:
                continue
            # 分数,标题,作者,原文句子,译文句子,赏析
            results = process_query(filename)
            results = results[:K]
            if len(results) == 0:
                continue
            total_p_data += len(results)
            # for c in content_list:
            #     for item in results:
            #         if c[0] == item[1] and c[1] == item[2]:
            #             fwashed.writerow(c)
            #             break
            # for c in content_list:
            #     for item in results:
            #         if c[2] == item[2][1][3] or (c[0] == item[2][1][1] and c[1] == item[2][1][2]):
            #             appear_number += 1
            #             total_right += 1
            #             break

            for item in results:
                for c in content_list:
                    if c[2] == item[2][1][3] or (c[0] == item[2][1][1] and c[1] == item[2][1][2]):
                        appear_number += 1
                        total_right += 1
                        break

            # recall = '%.6f' % (appear_number / len(content_list))
            precision = '%.6f' % (appear_number / len(results))
            total_pre += float(precision)
            temp_count += 1
            total_pre_count = '%.6f' % (total_pre / temp_count)
            # total_precision = '%.6f' % (total_right / total_p_data)
            print(f, "的精确度是：", precision)
            print("总的精确度是：", total_pre_count)
            # print("总的精确度是：", total_precision)
            dicname = root.split("/")
            flog.write(f + "：" + "召回率：" + precision + " 总召回率：" + total_pre_count + "\n")
            if appear_number == 0.000000:
                fw.write(f + "\n")
                # print("SBert对译文的分数：", item[0])
                # print("SBert对赏析的分数：", item[1])
                # print("文段的BM25分数：", item[2][1][0])
                # print("文段的关键词：", item[2][2])
                # print("原文：", item[2][1][3])
                # print("译文：", item[2][1][4])
                # print("赏析：", item[2][1][5])

    fw.close()
    flog.close()


if __name__ == '__main__':
    BM25Model, poetryList = get_corpus(corpus_path)
    get_all_transSentence(corpus_path)
    average_idf = sum(map(lambda k: float(BM25Model.idf[k]), BM25Model.idf.keys())) / len(BM25Model.idf.keys())
    # print("请开始输入，输入quit结束。")
    # query = input()
    # while query != "quit":
    #     process_query(query)
    #     print("请继续输入，输入quit结束：")
    #     query = input()
    test(QAdata_path, K=10)
