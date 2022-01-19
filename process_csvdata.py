# -*- coding:utf-8 -*-
import csv
from tqdm import tqdm
from match_analyze import choose_BM25, wash_analyze, Keyword_TextRank_TFIDF, get_TOPK_result
from Levenshtein import *
from util.utils import wash_content
from ltp import LTP

s1_path = "去重之后/总的去重合集/去重之后.csv"
s2_path = "去重之后/总的去重合集/分段补充.csv"
s3_path = "去重之后/总的去重合集/再次补全未分段内容.csv"
t1_path = "去重之后/总的去重合集/格式化数据（赏析分段+赏析去重）.csv"
# ['编号', '标题', '作者', '朝代', '原文', '翻译', '注释', '赏析', '标签']
t2_path = "去重之后/总的去重合集/诗级别的格式化数据（赏析分段+赏析去重+赏析与译文关键词LTP）.csv"
t3_path = "去重之后/分别提取的数据/译文.txt"
t4_path = "去重之后/总的去重合集/诗级别的格式化数据（赏析分段+赏析去重+赏析与译文关键词LTP）2.csv"
t5_path = "去重之后/总的去重合集/诗级别的格式化数据（赏析分段+赏析去重+赏析与译文关键词LTP）.csv"
test1_path = "去重之后/总的去重合集/测试.csv"
allTagpath = "去重之后/分别提取的数据/tag.txt"
tfidfPath = "../checkpoint/checkpoint_8_jieba/poetry_tfidf.model"
dicpath = "../checkpoint/checkpoint_8_jieba/poetry_dic.dict"
# ff = csv.reader(open(s3_path, 'r', encoding='utf-8'))
# ff = list(ff)


def merge_csv(file1, file2):  # file1是总的，file2是补全的分段的
    f1 = csv.reader(open(file1, 'r', encoding='utf-8'))
    f2 = csv.writer(open(file2, 'w+', encoding='utf-8'))
    # f2.writerow(['编号', '标题', '作者', '朝代', '原文', '翻译', '注释', '赏析'])

    for i in tqdm(f1):  # 对于file1中的每一首诗
        if len(i) == 0:
            continue
        flag = 0  # 标记是否有未分段的，1表示有。
        analyzes = i[7].split("|~|")
        analyzes = list(set(analyzes))
        if "" in analyzes:
            analyzes.remove("")
        if len(analyzes) == 0:  # 若没有赏析
            continue
        for a in analyzes:
            # print(a)
            if "|" not in a:  # 若有内容但是未分段
                flag = 1
                break

        if flag == 0:  # 若都分段了
            f2.writerow(i)
        elif flag == 1:  # 若存在未分段内容
            i[4] = "".join(i[4].split())
            # print("过去找的字段:", i[4])
            # j = search_content_s2(i[4])
            # if len(j) == 0:
            #     f2.writerow(i)
            # else:
            #     f2.writerow(j)


# merge_csv(t2_path, t3_path)


def calculate_csv(file):  # 统计数目
    shangxi, translation, annotation, st, sa, ta, sta = 0, 0, 0, 0, 0, 0, 0
    f = csv.reader(open(file, 'r', encoding='utf-8'))
    f_write = open("去重之后/总的去重合集/需要补全的.txt", 'w+', encoding='utf-8')
    count = 0
    for i in f:
        if len(i) == 0:
            continue

        alist = i[7].split("|~|")
        for aa in alist:
            aa.replace(" ", "")
            if aa != "" and "|" not in aa:
                # f_write.write(i[1] + " " + i[2] + "\n")
                print(i[1], " ", i[2])
                count += 1
                break

        analyze = i[7].replace("|~|", "").replace("|", "").replace("。", "").replace("，", "").replace(".", "").replace(
            "！", "").replace("？", "").replace(",", "").replace("\n", "")
        analyze = wash_content(analyze)
        if analyze != "":
            shangxi += 1
        trans = i[5].replace("|~|", "").replace("|", "").replace("。", "").replace("，", "").replace(".", "").replace("！",
                                                                                                                    "").replace(
            "？", "").replace(",", "").replace("\n", "")
        trans = wash_content(trans)
        if trans != "":
            translation += 1
        anno = i[6].replace("|~|", "").replace("|", "").replace("。", "").replace("，", "").replace(".", "").replace("！",
                                                                                                                   "").replace(
            "？", "").replace(",", "").replace("\n", "")
        anno = wash_content(anno)
        if anno != "":
            annotation += 1

        if analyze != "" and trans != "":
            st += 1
        if analyze != "" and anno != "":
            sa += 1
        if trans != "" and anno != "":
            ta += 1
        if trans != "" and anno != "" and analyze != "":
            sta += 1

    print("有赏析的:", shangxi)
    print("有翻译的:", translation)
    print("有注释的:", annotation)
    print("有赏析和翻译的:", st)
    print("有赏析和注释的:", sa)
    print("有译文和注释的:", ta)
    print("有赏析和注释和译文的:", sta)
    print("还是未分段的：", count)
    return shangxi


# calculate_csv(t4_path)


# '编号0', '标题1', '作者2', '朝代3', '原文4', '翻译5', '注释6', '赏析7'
# 构造格式化数据，诗中每个句子和其对应赏析被抽出来。
def set_dataformat(spath, tpath):
    f1 = csv.reader(open(spath, 'r', encoding='utf-8'))
    # f2 = csv.writer(open(tpath, 'w+', encoding='utf-8'))
    # f2.writerow(['标题', '作者', '朝代', '原文句子', '翻译句子', '注释', '关键段落', '全赏析'])
    for i in f1:  # 遍历每首诗
        if len(i) == 0:
            continue
        contents, translations, annotations, analyzes = i[4], i[5], i[6], i[7]
        clist, tlist = contents.split("|"), translations.split("|")  # 原文、译文是分句了的
        if "" in clist:
            clist.remove("")
        if "" in tlist:
            tlist.remove("")
        alist = analyzes.split("|")  # 赏析也有可能会由赏析、鉴赏等多个部分组成
        analyzelist, _ = wash_analyze(alist)
        if "" in analyzelist:
            analyzelist.remove("")
        analyzelist = list(set(analyzelist))
        if len(clist) == len(tlist) or len(tlist) == 0:
            for j in range(len(clist)):  # 若原文和译文是对齐的，遍历每一句话
                if len(tlist) == 0:
                    translation = ""
                    content = clist[j]
                else:
                    content, translation = clist[j], tlist[j]
                par = choose_BM25(content, translation, analyzelist)  # 计算每个段落的分数
                if len(par) >= 2:
                    if par[0][0] <= par[1][0] * 3:  # 如果第一、二段的分数在三倍以内
                        paragraph = par[0][1] + par[1][1]
                    else:
                        paragraph = par[0][1]
                else:
                    paragraph = par[0][1]
                # f2.writerow([i[1], i[2], i[3], content, translation, i[6], paragraph, analyzes])
                print(paragraph)
        else:  # 若原文和译文没有对齐，就以整首诗为单位
            par = choose_BM25(contents, translations, analyzelist)
            if len(par) >= 4:
                paragraph = par[0][1] + par[1][1] + par[2][1] + par[3][1] + par[4][1]
            else:
                paragraph = ""
                for item in par:
                    paragraph += item[1]
            # f2.writerow([i[1], i[2], i[3], contents, translations, i[6], paragraph, analyzes])
            print(paragraph)


# set_dataformat(t2_path, t2_path)


def refresh_content(path1, path2):
    f1 = csv.reader(open(path1, 'r', encoding='utf-8'))
    f2 = csv.writer(open(path2, 'w+', encoding='utf-8'))
    for i in f1:
        if len(i) == 0:
            continue
        analyzes = i[7]
        new_analyzes = []
        alist = analyzes.split("|~|")
        if "" in alist:
            alist.remove("")
        alist = list(set(alist))
        for a in alist:
            if a not in new_analyzes and "|" in a:  # 如果该条赏析分段了，且不重复
                new_analyzes.append(a)
        if len(new_analyzes) == 0:  # 如果经过上述操作，没有合适的赏析了
            best_analyze = ""
            for a in alist:
                if len(a) > len(best_analyze):
                    best_analyze = a
            i[7] = best_analyze
        else:
            analyzes_content = ""
            for item in new_analyzes:
                analyzes_content += (item + "|~|")
            i[7] = analyzes_content
        f2.writerow(i)


# refresh_content(t2_path, t3_path)


# 获取所有的译文
def get_all_translations(spath, tpath):
    f1 = csv.reader(open(spath, 'r', encoding='utf-8'))
    f2 = open(tpath, 'w+', encoding='utf-8')
    for i in f1:
        if len(i) == 0 or i[0] == '编号' or i[5] == "":
            continue
        f2.write(i[5] + "\n")
    f2.close()


# get_all_translations(t2_path, t3_path)


def wash_csvAnalyze(path1, path2):
    f1 = csv.reader(open(path1, 'r', encoding='utf-8'))
    f2 = csv.writer(open(path2, 'w+', encoding='utf-8'))
    for i in f1:
        if len(i) == 0:
            continue
        analyzes = i[7].split("|~|")
        if "" in analyzes:
            analyzes.remove("")
        alist = []
        for item in analyzes:
            paragraphs = item.split("|")
            if "" in paragraphs:
                paragraphs.remove("")
            for paragraph in paragraphs:
                alist.append(paragraph)
        if '' in alist:
            alist.remove('')
        alist = list(set(alist))
        new_analyzes = ""
        for j in range(len(alist)):
            flag = 0
            for k in range(len(alist)):
                if k == j:
                    continue
                if alist[j] in alist[k]:    # 若alist[j]被除了自己的其它内容包含。
                    flag = 1
                    break
            if flag == 0:   # 若没有被包含，则可以要。
                new_analyzes += (alist[j] + "|")
        i[7] = new_analyzes
        f2.writerow(i)


# wash_csvAnalyze(t1_path, t2_path)


def get_poetry_tag(spath, tpath):   # 给诗歌添加标签
    f1 = csv.reader(open(spath, 'r', encoding='utf-8'))
    f2 = csv.writer(open(tpath, 'w+', encoding='utf-8'))
    f2.writerow(['编号', '标题', '作者', '朝代', '原文', '翻译', '注释', '赏析', '标签'])
    ltp = LTP(path="base")
    for i in f1:
        if len(i) == 0 or i[0] == '编号':
            continue
        i[1], i[2], i[3] = i[1].replace(",", "，"), i[2].replace(",", "，"), i[3].replace(",", "，")
        i[4], i[5], i[6], i[7] = i[4].replace(",", "，"), i[5].replace(",", "，"), i[6].replace(",", "，"), \
                                 i[7].replace(",", "，")
        token_states = Keyword_TextRank_TFIDF("", "", i[5] + i[7], ltp)
        tags = ""
        for token in token_states:
            tags += (token[0] + "，")
        # alist, nlist, tlist, vlist, slist, clist = get_TOPK_result(token_states)
        # nlength = len(nlist)
        # if nlength * 0.3 > 40:
        #     kn = 40
        # else:
        #     kn = int(nlength * 0.3)
        #
        # vlength = len(vlist)
        # if vlength * 0.3 > 40:
        #     kv = 40
        # else:
        #     kv = int(vlength * 0.3)
        #
        # tags = ""
        # for token in alist:
        #     tags += (token[0] + "，")
        # for token in nlist[:kn]:
        #     tags += (token[0] + "，")
        # for token in tlist:
        #     tags += (token[0] + "，")
        # for token in vlist[:kv]:
        #     tags += (token[0] + "，")
        # for token in slist:
        #     tags += (token[0] + "，")
        # for token in clist:
        #     tags += (token[0] + "，")
        i.append(tags)
        f2.writerow(i)


# get_poetry_tag(t1_path, t4_path)


def get_allTags(spath, tpath):      # 获取所有的tag
    fr = csv.reader(open(spath, "r", encoding="utf-8"))
    fw = open(tpath, "w+", encoding="utf-8")
    token_list = []
    for i in fr:
        if len(i) == 0:
            continue
        tokens = i[8].split("，")
        for token in tokens:
            token_list.append(token)
    token_list = list(set(token_list))
    for token in token_list:
        fw.write(token + "\n")
    fw.close()


# get_allTags(t2_path, allTagpath)


def cut_duplicate(spath, tpath):    # 消除csv中重复的数据
    fr = csv.reader(open(spath, "r", encoding="utf-8"))
    fw = csv.writer(open(tpath, "w+", encoding="utf-8"))
    item_list = []
    for i in fr:
        flag = 0
        if len(i) == 0 or i[0] == '编号':
            continue
        i[4] = i[4].replace("\u3000", "")
        i[5] = i[5].replace("\u3000", "")
        i[6] = i[6].replace("\u3000", "")
        i[7] = i[7].replace("\u3000", "")
        for item in item_list:
            if distance(i[4][:9], item[4][:9]) <= 1:    # 有重复的
                flag = 1
                break
        if flag == 0:
            item_list.append(i)
    fw.writerow(['编号', '标题', '作者', '朝代', '原文', '翻译', '注释', '赏析', '标签'])
    for item in item_list:
        fw.writerow(item)


# cut_duplicate(t4_path, t5_path)


def find_word_InTags(spath, tpath, keyword):
    fr = csv.reader(open(spath, "r", encoding="utf-8"))
    fw = csv.writer(open(tpath, "w+", encoding="utf-8"))
    for i in fr:
        if len(i) == 0:
            continue
        tags = i[8].split("，")
        if keyword in tags:
            print("发现：", keyword)
            fw.writerow(i)


# find_word_InTags(t2_path, test1_path, "睡觉")


# 清洗停用词表
def cut_duplicate_stopwords(spath, tpath):
    fs = open(spath, "r", encoding="utf=8")
    fw = open(tpath, "w+", encoding="utf-8")
    lines = fs.readlines()
    tokens = []
    for line in lines:
        line = line.replace("\n", "")
        tokens.append(line)
    tokens = list(set(tokens))
    for token in tokens:
        fw.write(token + "\n")
    fs.close()
    fw.close()


# spath = "stopwords.txt"
# tpath = "stopword.txt"
# cut_duplicate_stopwords(spath, tpath)


def calculate_csv_ltp(path):
    f = csv.reader(open(path, "r", encoding="utf-8"))
    for i in f:
        if len(i) == 0 or i[0] == '编号':
            continue
        if len(i) > 9:
            print(len(i))


# calculate_csv_ltp(t2_path)
