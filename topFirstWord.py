#coding=utf-8
# 获取该小说开头的n个字符
import os
import collections
import sys
o_path = os.getcwd()
# print(o_path)
# sys.path.append("..")
from inference import zhetian
'''
返回前150个开头字符
'''
def process_poems(file_name):
    # 诗集
    poems = []
    print('file_name:', file_name, "\n")
    # 这一段循环读取数据集，返回诗句的内容，按理说内容中不会有空格了，然后首尾加上G和E
    # 'G偶避蝉声来隙地，忽随鸿影入辽天。闲僧不会寂寥意，道学西方人坐禅。E'
    with open(file_name, "r", encoding='utf-8', ) as f:
        # 这里明明应该用 readline()的
        i = 0
        for line in f.readlines():
            if len(line) == 1:
                continue
            if '_' in line or '(' in line or '(' in line or '《' in line or '[' in line or \
                 '"' in line or '“' in line or '”' in line or '<' in line or '>' in line\
                    or 'b' in line or 'o' in line or 's' in line or '（' in line or '）' in line or 'y' in line:
                continue
            for word in line:
                if  ' ' == word or '“' == word or '”' == word or '？' == word or '?' == word or '！' == word \
                        or '!' == word or '。' == word or '.' == word or '‘' == word or '，' == word:
                    continue
                else:
                    firstWord = word
                    poems.append(firstWord)
                    break
    counter = collections.Counter(poems)
    count_pairs = sorted(counter.items(), key=lambda  x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[0:150]
    # print(len(words ))
    # for word in words:
    #     print(word)
    # words, _ = zip(*counter)
    return words


if __name__ == '__main__':
    file_name = os.path.abspath('./dataset/data/zhetian.txt')
    write_file = os.path.abspath('./dataset/data/')
    words = process_poems(file_name)

    for i in range(24, 48, 6) :
        write_file = write_file + str(i)

        for word in words:
            poem2 = zhetian.gen_poem('叶', 24)
            print(poem2)
            exit()
            # with open(file_name, "w", encoding='utf-8', ) as f:
            #     f.write(poem2+"\n")
            #     exit()