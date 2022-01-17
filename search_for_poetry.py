# coding:utf-8
from gensim import corpora
from gensim.models import TfidfModel, LdaModel
from model_TFIDF_LDA import get_stopwords
from nltk.corpus import wordnet as wn
from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import gensim
from gensim.models import KeyedVectors
import jieba.posseg as psg
from ltp import LTP
import functools
import synonyms
from util.utils import normalization, compare_right, compare_left, wash_list, wash_content


# 可接受词性 jieba
# flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an', 'ng', 'tg', 'vg', 'vd', 's', 'i', 'l']
flags = ['c', 'e', 'm', 'nh', 'o', 'p', 'r', 'u', 'wp', 'ws', 'x', 'd']  # 停止词性 LTP
# ['编号', '标题', '作者', '朝代', '原文', '翻译', '注释', '赏析', '标签']
corpus_path = "去重之后/总的去重合集/诗级别的格式化数据（赏析分段+赏析去重+赏析与译文关键词LTP）.csv"
embedding_path = "wordEmbedding/sgns.baidubaike.bigram-char"
tfidfPath = "checkpoint/checkpoint_100topics_50epoch/poetry_tfidf.model"
dicpath = "checkpoint/checkpoint_100topics_50epoch/poetry_dic.dict"
ldapath = "checkpoint/checkpoint_100topics_50epoch/poetry_lda.model"
allTagpath = "去重之后/分别提取的数据/tag.txt"
stop_path = "stopwords.txt"
sentence_model = SentenceTransformer('hfl/chinese-bert-wwm-ext')
stopwords = get_stopwords(stop_path)
tfidfModel = TfidfModel.load(tfidfPath)
ldaModel = LdaModel.load(ldapath)
dictionary = corpora.Dictionary.load(dicpath)
model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
words = model.vocab
words_list = [word for i, word in enumerate(words)]  # 词向量文件中的词列表
ltp = LTP(path="base")


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
    for i in f:
        if len(i) == 0 or i[0] == '编号':
            continue
        tags = i[8].split("，")
        corpus.append(tags)
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


def calculate_BM25_matchingScores(contents, translations, analyze_list):
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
        poetry_contents = content.split("。")  # 尝试手动分句
        trans_contents = translation.split("。")
        poetry_contents, trans_contents = wash_list(poetry_contents), wash_list(trans_contents)
        if len(poetry_contents) == 1:  # 若分完还是一句。
            query_list = []
            for qtt in qs:
                if qtt in analyze or qtt in content or qtt in translation:
                    query_list.append(qtt)
            return [(content, translation, query_list)]
        elif len(poetry_contents) != len(trans_contents) and len(poetry_contents) > 1:  # 长度大于1，但是不对齐。
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, [], analyzelist)
        else:  # 若手动分完后是对齐的了...
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, trans_contents, analyzelist)
    elif len(contents) == len(translations):  # 本来就对齐
        sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(contents, translations, analyzelist)
    else:  # 若不对齐
        poetry_contents = content.split("。")  # 先尝试手动分句
        trans_contents = translation.split("。")
        poetry_contents, trans_contents = wash_list(poetry_contents), wash_list(trans_contents)
        if len(poetry_contents) == 1:  # 若手动分句后只有一句
            query_list = []
            for qtt in qs:
                if qtt in analyze or qtt in content or qtt in translation:
                    query_list.append(qtt)
            return [(content, translation, query_list)]
        elif len(poetry_contents) != len(trans_contents):  # 若手动分句后有多句，但是不对齐，则只用原文去搜
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, [], analyzelist)
        else:  # 分句后对齐了
            sentenceAnalyzes, scoreDict = calculate_BM25_matchingScores(poetry_contents, trans_contents, analyzelist)
    # scoreDict：{赏析的段落：[[分数、原文、译文], [分数、原文、译文]...]}，sentenceAnalyzes：[[sentence, trans, paragraph],...]
    # 首先通过是否出现在原文、译文的方式找关键句。

    maxCount = 0
    maxSentence = []    # [(score, content, translation, [q1, q2, q3])]
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
            maxSentence = [(value[0][0], value[0][1], value[0][2], query_list, key)]
            if len(value) >= 2 and value[0][0] <= value[1][0] + 0.5 and len(value[0][1]) >= 7:
                maxSentence.append((value[1][0], value[1][1], value[1][2], query_list, key))
            elif len(value[0][1]) < 7:  # 太短了也自动扩充
                if value[0][1][-1] != "。" or value[0][1][-1] != "，" or value[0][1][-1] != "！" or value[0][1][-1] != "？":
                    maxSentence = [
                        (max(value[0][0], value[1][0]), value[0][1] + "。" + value[1][1], value[0][2] + value[1][2],
                         query_list, key)]
                else:
                    maxSentence = [(max(value[0][0], value[1][0]), value[0][1] + value[1][1], value[0][2] + value[1][2],
                                    query_list, key)]
    if maxSentence[0][0] < 3:
        return []
    return maxSentence  # 函数返回的是包含了最多检索词的赏析所在的诗句


# 对搜索到的所有诗的结果进行排名。mode=1：若结果中包含了原始的、扩充前的query，则排在前面。mode=2：若结果包含了分别被扩充的多个query，则排在前面。
def get_result(query_context, qs, aidf, old_query, query_set_list, mode=0):  # mode为0则不需要将包含了old_query的排在前面
    scores = BM25Model.get_scores(qs, aidf)  # 输入的文本内容必须是由关键词组成的列表
    paragraph_score = []
    for i in range(len(scores)):  # 分数，标题，作者，朝代，原文，翻译，注释，赏析，标签
        paragraph_score.append((scores[i], poetryList[i][1], poetryList[i][2], poetryList[i][3], poetryList[i][4],
                                poetryList[i][5], poetryList[i][6], poetryList[i][7], poetryList[i][8]))
    paragraph_score.sort(key=functools.cmp_to_key(compare_left))
    paragraph_score = [item for item in paragraph_score[:25] if item[0] > 0]
    print("获取到：", len(paragraph_score), "条结果。")
    paragraph_score_new = []
    # 几种mode都是在对诗进行排名。
    if mode == 1:  # 将包含了old_query的排在前面
        for item in paragraph_score:
            for q1 in old_query:
                if q1 in item[4] or q1 in item[5] or q1 in item[7]:
                    paragraph_score_new.append(item)
                    break
        for item in paragraph_score:
            flag = 0
            for q2 in old_query:
                if q2 in item[4] or q2 in item[5] or q2 in item[7]:
                    flag = 1
                    break
            if flag == 0:
                paragraph_score_new.append(item)
    elif mode == 0:
        paragraph_score_new = paragraph_score
    elif mode == 2:  # query_set_list: {q1:[q1,...], q2:[q2,...], q3:[q3,...]}，根据出现在用户输入中的词的数量
        paragraph_score_count = []
        keys = query_set_list.keys()
        for item in paragraph_score:
            count = 0
            poetry_keyword = []
            for key in keys:
                q_set_list = query_set_list[key]
                for qq in q_set_list:
                    if qq in item[4] or qq in item[5] or qq in item[7]:
                        count += 1  # 统计同时是q1,q2,q3的列表中的词语
                        poetry_keyword.append(qq)
                        break
            paragraph_score_count.append([count, item, poetry_keyword])
        paragraph_score_count.sort(key=functools.cmp_to_key(compare_left))
        left, right = 0, 0  # 算法：对于count相同的item，拥有old_query较多的优先排在前面。
        paragraph_new = []
        while right <= len(paragraph_score_count):
            if right == len(paragraph_score_count) or paragraph_score_count[left][0] != paragraph_score_count[right][0]:
                temps = paragraph_score_count[left:right]
                temps_count = []
                for temp in temps:
                    count = 0
                    for qb in old_query:
                        if qb in temp[1][4] or qb in temp[1][5] or qb in temp[1][7]:
                            count += 1
                    temps_count.append([count, temp])
                temps_count.sort(key=functools.cmp_to_key(compare_left))
                paragraph_new.extend(temps_count)
                if right == len(paragraph_score_count):
                    break
                left = right
            elif paragraph_score_count[left][0] == paragraph_score_count[right][0]:
                right += 1
        for item in paragraph_new:
            print(item)
            paragraph_score_new.append([item[1][1], item[1][2]])
    qs_string = '，'.join(qs)
    query_embeddings = sentence_model.encode([query_context])[0]
    for i in range(len(paragraph_score_new)):   # 遍历每首诗
        poetry_item = paragraph_score_new[i][0]  # 分数，标题，作者，朝代，原文，翻译，注释，赏析，标签
        poetry_lda_score = LDA_sim(qs_string, poetry_item[7])
        # [(score, content, translation, [q1, q2, q3], analyze), ...]
        sentence_result = sentence_fromPoetry(poetry_item, qs)
        if not sentence_result:
            continue
        print("诗歌：", poetry_item[1], " 作者：", poetry_item[2])
        for item in sentence_result:
            ldaScore = LDA_sim(qs_string, item[4])
            context_embeddings = sentence_model.encode([item[2] + item[4]])[0]
            cos = cosine_similarity([query_embeddings], [context_embeddings])
            print("整诗的BM25分数:", poetry_item[0], " Bert分数:", cos[0][0],
                  " 整诗的LDA分数:", poetry_lda_score, " 对应文段的LDA分数:", ldaScore)
            print("原文与赏析的BM25分数:", item[0])
            print("整诗的关键词：", paragraph_score_new[i][1])
            print("文段的关键词：", item[3])
            print("原文：", item[1])
            print("译文：", item[2])
            print("赏析：", item[4])
            print("\n")
    return paragraph_score_new


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
            if len(q_expansion) == 2:
                query_list.extend(q_expansion)
                if k == 1:
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
                else:   # doesn't match的刚好为q，则直接删除最后一个。
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
    results = model.most_similar(token, topn=100)
    for item in results:
        if item[0] in tagList:
            print("word2vec替换结果：", item[0])
            re_token = item[0]
            break
    if re_token == "":
        print("word2vec替换失败。")
    return re_token


def replace_synonyms(qlist, qSet_dict):  # 遍历分词后的word列表，并将不在数据集的词表中的词语换成数据集中的近义词
    queryList = []
    for token in qlist:
        if token not in tagList:
            print("需要替换：", token)
            qSet_dict.pop(token)
            if token not in words_list:
                print("LTP分词后无法被word2vec替换：", token)
                if len(token) > 2:  # 词长大于2时才考虑重新分词。
                    token_jieba = psg.cut(token)
                    tagFlag = 0
                    for word, flag in token_jieba:
                        if word in tagList:  # 如果刚好在tagList，则该词直接替换完成。
                            print("JIEBA分词后，可以直接使用：", word)
                            queryList.append(word)
                            qSet_dict[word] = [word]
                            tagFlag = 1
                        if word not in tagList and word in words_list:  # 若不在tagList但是在word2vec中。
                            re_token = replace_with_word2vec(word)
                            print("JIEBA分词后，可以用word2vec替换：", word, " 为：", re_token)
                            queryList.append(re_token)
                            qSet_dict[re_token] = [re_token]
                            tagFlag = 1
                    if tagFlag == 1:
                        continue
                    # 若没有被换掉，再看一次是否在word2vec中。
                    print("JIEBA分词后不可以直接或间接替换。")

            # if count == 3 or (count == 2 and item[1] < 0.5):    # 多拓展1或2个
            #     break
            # if item[0] in tagList:
            #     print("替换结果：", item[0])
            #     queryList.append(item[0])
            #     qSet_dict[token].append(item[0])
            #     count += 1
            # break
            else:
                re_token = replace_with_word2vec(token)
                print("直接使用word2vec将：", token, " 替换成：", re_token)
                queryList.append(re_token)
                qSet_dict[re_token] = [re_token]
        else:  # 不需要被替换
            queryList.append(token)
            qSet_dict[token] = [token]
    return queryList, qSet_dict


def process_query(context):
    seg, hidden = ltp.seg([context])
    seg = seg[0]
    pos = ltp.pos(hidden)[0]
    qs = []
    for i in range(len(seg)):  # 对输入的query进行分词，并去除停用词
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
    # if len(qs) > 3:
    #     qs = cut_query(qs, k=3)
    print("初步检索的结果：")
    print(query_set_dict)
    # 进行搜索
    paragraph_score = get_result(context, qs, average_idf, old_qs, query_set_dict, mode=2)
    # 自动扩充搜索范围
    if len(paragraph_score) == 0 or paragraph_score[0][0] <= 8:  # 若没有获取到结果，或者分数都不高
        if len(qs) > 5:  # 如果关键词太多，则需要取重要的
            # qs = cut_query(context, qs, k=5)
            qs, query_set_dict = expanding_query_withDeleting(qs, query_set_dict, 3)
            print("扩展后")
            print(query_set_dict)
        elif len(qs) == 1:
            qs, query_set_dict = expanding_query_withDeleting(qs, query_set_dict, 6)
        else:
            qs, query_set_dict = expanding_query_withDeleting(qs, query_set_dict, 4)
        print("扩充后再次检索：")
        new_paragraph_score = get_result(context, qs, average_idf, old_qs, query_set_dict, mode=2)


if __name__ == '__main__':
    BM25Model, poetryList = get_corpus(corpus_path)
    average_idf = sum(map(lambda k: float(BM25Model.idf[k]), BM25Model.idf.keys())) / len(BM25Model.idf.keys())
    print("请开始输入，输入quit结束。")
    query = input()
    while query != "quit":
        process_query(query)
        print("请继续输入，输入quit结束：")
        query = input()
