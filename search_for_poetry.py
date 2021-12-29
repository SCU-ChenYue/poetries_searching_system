# coding:utf-8
from gensim import corpora
from gensim.models import TfidfModel
from model_TFIDF_LDA import get_stopwords
from nltk.corpus import wordnet as wn
from gensim.summarization import bm25
import jieba
import csv
from gensim.models import KeyedVectors
import jieba.posseg as psg
from ltp import LTP
import jieba.analyse
import functools
import synonyms
from util.utils import normalization, compare_right, compare_left, cut_sent

# 可接受词性 jieba
# flags = ['a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an', 'ng', 'tg', 'vg', 'vd', 's', 'i', 'l']
# 停止词性 LTP
flags = ['c', 'e', 'm', 'nh', 'o', 'p', 'r', 'u', 'wp', 'ws', 'x']
corpus_path = "去重之后/总的去重合集/诗级别的格式化数据（赏析分段+赏析去重+关键词LTP）.csv"  # ['编号', '标题', '作者', '朝代', '原文', '翻译', '注释', '赏析', '标签']
embedding_path = "wordEmbedding/sgns.baidubaike.bigram-char"
tfidfPath = "checkpoint/checkpoint_9/poetry_tfidf.model"
dicpath = "checkpoint/checkpoint_9/poetry_dic.dict"
allTagpath = "去重之后/分别提取的数据/tag.txt"
stop_path = "stopwords.txt"
stopwords = get_stopwords(stop_path)
tfidfModel = TfidfModel.load(tfidfPath)
dictionary = corpora.Dictionary.load(dicpath)
model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
words = model.vocab
words_list = [word for i, word in enumerate(words)]  # 词向量文件中的词列表
ltp = LTP(path="small")


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


tagList = get_all_tags()    # 所有的诗歌标签


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
    return analyze_list, newcontent    # 由每个段落组成的list [...,...]


def cut_query(context, q, k=5):  # 原输入句子和去停用词后的词语列表
    token_score = {}
    for item in q:
        token_score[item] = 0

    # 先使用语料库中的tfidf进行打分
    corpus = [dictionary.doc2bow(q)]
    corpus_tfidf = list(tfidfModel[corpus])[0]
    corpus_tfidf.sort(key=functools.cmp_to_key(compare_right))
    token_state = []
    for item in corpus_tfidf:
        word = dictionary[item[0]]
        token_state.append((word, item[1]))
    print("语料库的tf-idf打分情况：", token_state)

    for item in token_state:
        token_score[item[0]] = item[1]
    print("被补充前的token_score：", token_score)
    extra_words = []
    for word, score in token_score.items():
        if score == 0:
            extra_words.append(word)
    if len(extra_words) == 0:
        return

    # 再用自带的tf-idf和text-rank打分，用于补充
    jieba.analyse.set_stop_words(stop_path)
    # qr = jieba.analyse.textrank(context, topK=20, withWeight=True,
    #                             allowPOS=('a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an',
    #                                       'ng', 'tg', 'vg', 'vd', 's', 'i'))
    # print("text-rank重要性解析结果：", qr)
    qt = jieba.analyse.extract_tags(context, topK=20, withWeight=True,
                                    allowPOS=('a', 'j', 'n', 'vn', 'ns', 't', 'v', 's', 'ad', 'ag', 'an',
                                              'ng', 'tg', 'vg', 'vd', 's', 'i'))
    print("归一化前的tf-idf重要性解析结果：", qt)
    if len(qt) != 0:
        qt = tfidf_normalization(qt)
    print("归一化后的tf-idf重要性解析结果：", qt)

    for word in extra_words:
        for item in qt:
            if word == item[0]:
                token_score[word] = item[1]
                break
    token_score = sorted(token_score.items(), key=lambda d: d[1], reverse=True)
    print("被补充、排序后的token_score：", token_score)
    sorted_tokens = []
    for word, score in token_score[:k]:
        sorted_tokens.append(word)
    return sorted_tokens


def get_result(qs, aidf):
    scores = BM25Model.get_scores(qs, aidf)  # 输入的文本内容必须是由关键词组成的列表
    paragraph_score = []
    for i in range(len(scores)):    # 分数，标题，作者，朝代，原文，翻译，注释，赏析，标签
        paragraph_score.append((scores[i], poetryList[i][1], poetryList[i][2], poetryList[i][3], poetryList[i][4],
                                poetryList[i][5], poetryList[i][6], poetryList[i][7], poetryList[i][8]))
    paragraph_score.sort(key=functools.cmp_to_key(compare_left))
    for item in paragraph_score[:25]:
        if item[0] == 0:
            break
        print("全诗信息：")
        print(item[0], " ", item[1], " ", item[2], " ", item[3], " ", item[4])
        print("目标句子提取结果：")
        sentence_result = sentence_fromPoetry(item, qs)
        print(sentence_result)
    return paragraph_score


def search_multiSentence_analyze(contents, translations, analyze_list):
    sentence_analyze = []
    for i in range(len(contents)):  # 若原文和译文对齐，则依次遍历每一句话。
        sentence = contents[i]
        if len(translations) == 0:  # 有可能并没有传进来翻译
            trans = ""
        else:
            trans = translations[i]
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

        aidf = sum(map(lambda m: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
        poetry_query = ltp.seg([sentence + trans])
        scores = bm25Model.get_scores(poetry_query, aidf)  # 搜索得到该句话对应每个文段的分数
        paragraphs_score = []

        for j in range(len(scores)):
            paragraphs_score.append((scores[j], analyze_list[j]))
        paragraphs_score.sort(key=functools.cmp_to_key(compare_left))
        if paragraphs_score[0][0] <= paragraphs_score[1][0] * 3:  # 如果第一、二段的分数在三倍以内，则两段都要
            paragraph = paragraphs_score[0][1] + paragraphs_score[1][1]
        else:
            paragraph = paragraphs_score[0][1]
        sentence_analyze.append([sentence, trans, paragraph])
    return sentence_analyze


def sentence_fromPoetry(poetry_item, qs):  # 对于一首诗，利用原文、译文，从赏析中检索对应的文段。
    content, translation, analyze = poetry_item[4], poetry_item[5], poetry_item[7]
    content, translation, analyze = content.replace(" ", ""), translation.replace(" ", ""), analyze.replace(" ", "")
    sentenceAnalyzes = []   # [[sentence, trans, paragraph],...]
    # 对赏析进行分段，并再分成短句后进行清洗。
    analyzes = analyze.split("|")
    analyze_list, _ = wash_analyze(analyzes)
    if "\n" in analyzes:
        analyzes.remove("\n")
    if "" in analyzes:
        analyzes.remove("")
    contents, translations = content.split("|"), translation.split("|")
    if "" in contents:
        contents.remove("")
    if "" in translations:
        translations.remove("")

    if len(contents) == 1:  # 如果默认分句的结果只有一句
        poetry_contents = content.split("。")    # 尝试手动分句
        trans_contents = translation.split("。")
        if len(poetry_contents) == 1 or len(poetry_contents) != len(trans_contents):  # 若分完还是一句或没法对齐。
            return [content, translation, analyze]
        else:   # 若手动分完后是对齐的了...
            sentenceAnalyzes = search_multiSentence_analyze(poetry_contents, trans_contents, analyze_list)
    elif len(contents) == len(translations):    # 本来就对齐
        sentenceAnalyzes = search_multiSentence_analyze(contents, translations, analyze_list)
    else:   # 若不对齐
        poetry_contents = content.split("。")    # 先尝试手动分句
        trans_contents = translation.split("。")
        if len(poetry_contents) == 1:    # 若手动分句后只有一句
            return [content, translation, analyze]
        elif len(poetry_contents) != len(trans_contents):   # 若手动分句后有多句，但是不对齐，则只用原文去搜
            sentenceAnalyzes = search_multiSentence_analyze(poetry_contents, [], analyze_list)

    maxCount = 0
    maxItem = []
    for item in sentenceAnalyzes:
        count = 0
        for q in qs:
            if q in item[1] or q in item[2]:
                count += 1
        if count > maxCount:
            maxCount = count
            maxItem = item
    return maxItem  # 函数返回的是包含了最多检索词的赏析所在的诗句


def expanding_query_withDeleting(q_list, k):
    print("进行扩充，每个词扩充", k, "个。")
    query_list = q_list.copy()
    for q in q_list:
        print("开始扩充", q)
        q_expansion = [q]   # 对于q，自己有一个扩充列表
        print("开始wordnet扩充。")
        wordnets = []
        wordnetSynset = wn.synsets(q, lang='cmn')
        for synset in wordnetSynset:
            wordnets.extend(synset.lemma_names('cmn'))
        wordnets = list(set(wordnets))
        if len(wordnets) >= 1:
            for i in range(len(wordnets)):
                wordnets[i] = wordnets[i].replace("+", "")
                if len(q_expansion) >= k + 2:  # 最多为k+2个
                    break
                if wordnets[i] in tagList and wordnets[i] not in query_list and wordnets[i] not in q_expansion:
                    q_expansion.append(wordnets[i])
            if len(q_expansion) == 2:  # 只扩充了一个则不删除
                query_list.extend(q_expansion)
                if k == 1:  # 若已经够了
                    print(q, "的最终扩充结果为：", q_expansion)
                    continue
            elif len(q_expansion) >= 3:
                different_token = model.doesnt_match(q_expansion)
                if different_token != q:
                    print("删除：", different_token)
                    q_expansion.remove(different_token)
                else:
                    print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                if len(q_expansion) == k + 1:  # 若已经够了
                    query_list.extend(q_expansion)
                    print(q, "的最终扩充结果为：", q_expansion)
                    continue
        else:
            print("wordnet未获取到内容！")
        print("开始synonyms扩充。")
        synonyms_list = synonyms.nearby(q, 50)
        synonyms_words = synonyms_list[0]
        synonyms_scores = synonyms_list[1]
        if len(synonyms_words) >= 1:    # 至少得有内容
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
                    continue
            elif len(q_expansion) >= 3:
                different_token = model.doesnt_match(q_expansion)
                if different_token != q:
                    print("删除：", different_token)
                    q_expansion.remove(different_token)
                else:
                    print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                if len(q_expansion) == k + 1:  # 若已经够了
                    query_list.extend(q_expansion)
                    print(q, "的最终扩充结果为：", q_expansion)
                    continue
        else:
            print("synonyms未获取到内容！")
        print("开始word2vec扩充。")
        if q not in words_list:
            print("word2vec未获取到内容！")
            print(q, "的最终扩充结果为：", q_expansion)
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
                if different_token != q:
                    print("删除：", different_token)
                    q_expansion.remove(different_token)
                else:
                    print("删除：", q_expansion[-1])
                    q_expansion.remove(q_expansion[-1])
                query_list.extend(q_expansion)
        print(q, "的最终扩充结果为：", q_expansion)
    query_list = list(set(query_list))
    return query_list


def replace_synonyms(qlist):    # 遍历分词后的word列表，并将不在数据集的词表中的词语换成数据集中的近义词
    queryList = []
    for token in qlist:
        if token not in tagList:
            print("需要替换：", token)
            count = 0
            if token not in words_list:
                print("无法替换：", token)
                continue
            print("正在通过word2vec获取100个近义词。")
            result_list = model.most_similar(token, topn=100)
            print("获取完成。")
            for item in result_list:
                if count == 3:
                    break
                if item[0] in tagList:
                    print("替换结果：", item[0])
                    queryList.append(item[0])
                    count += 1
                    # break
        else:
            queryList.append(token)
    return queryList


def process_query(context):
    seg, hidden = ltp.seg([context])
    seg = seg[0]
    pos = ltp.pos(hidden)[0]
    qs = []
    for i in range(len(seg)):
        word = seg[i].strip()
        flag = pos[i].strip()
        if word not in stopwords and flag not in flags:
            qs.append(word)
    # query = psg.cut(context)
    # qs = []
    # for word, flag in query:
    #     if word not in stopwords and flag in flags:
    #         qs.append(word)
    qs = list(set(qs))
    print("去掉停用词和停用词性后的结果：", qs)
    qs = replace_synonyms(qs)   # 替换不在词表中的词
    print("进行近义词替换后的结果：", qs)
    if len(qs) <= 2:    # 若此时词语很少
        # 进行词语扩充
        qs = expanding_query_withDeleting(qs, k=3)

    # if len(qs) > 5:
    #     qs = cut_query(context, qs, k=5)

    print("初步检索的结果：")
    paragraph_score = get_result(qs, average_idf)

    # 自动扩充搜索范围
    if len(paragraph_score) == 0 or paragraph_score[0][0] <= 8:  # 若没有获取到结果，或者分数都不高
        if len(qs) > 5:  # 如果关键词太多，则需要取重要的
            # qs = cut_query(context, qs, k=5)
            qs = expanding_query_withDeleting(qs, 3)
        elif len(qs) == 1:
            qs = expanding_query_withDeleting(qs, 6)
        else:
            qs = expanding_query_withDeleting(qs, 4)
        print("扩充后再次检索：")
        new_paragraph_score = get_result(qs, average_idf)


if __name__ == '__main__':
    BM25Model, poetryList = get_corpus(corpus_path)
    average_idf = sum(map(lambda k: float(BM25Model.idf[k]), BM25Model.idf.keys())) / len(BM25Model.idf.keys())
    print("请开始输入，输入quit结束。")
    query = input()
    while query != "quit":
        process_query(query)
        print("请继续输入，输入quit结束：")
        query = input()
