#coding=utf-8

import collections
import os
import sys
import numpy as np

start_token = 'G'
end_token = 'E'

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
            # 如果只有一个符号，那么跳过该行
            if len(line) == 1:
                continue
            # print('line...')
            # print(line)
            # if  i > 5 :
            #     exit()
            try:
                content = line.replace(' ', '')
                content = content.replace("\n", '')
                # 处理掉特殊字符，直接跳过，此处把对话的内容跳过
                if '_' in content or '(' in content or '(' in content or '《' in content or '[' in content or \
                    start_token in content or end_token in content or '"' in content or '“' in content or '”' in content:
                    continue
                # 如果内容太小或者太大，都不看了
                if len(content) < 5 or len(content) > 120:
                    continue
                # 将诗的内容拼接把首尾字符添加
                content = start_token + content + end_token
                # 将内容添加进去
                # print('content:', content, "\n")
                poems.append(content)
            except ValueError as e:
                pass

    poems = sorted(poems, key=lambda l: len(line))
    # print(poems)
    # exit()
    # 统计每个字出现次数
    # all_words 中是所有的出现过的字
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里获取每个字对应的频率
    # 这里可以尝试把偏僻字的诗直接剔除
    # collections的Counter可以为hashable对象计数
    '''
    counter
    {'，': 123222, '。': 122489, 'G': 34692, 'E': 34692, '不': 13522, '人': 12015, '山': 10170, '风': 9680, '日': 8890,
     '无': 8586, '一': 8547, '云': 7898, '花': 7436, '春': 7396, '来': 7326, '何': 6874, '水': 6864, '月': 6821, '上': 6646,
      '有': 6611, '时': 6483, '中': 6358, '天': 6135, '年': 5970, '归': 5524, '秋': 5412, '相': 5283, '知': 5259, '长': 5141, 
      '去': 4959, '自': 4947, '君': 4922, '心': 4896, '夜': 4858, '江': 4854, '生': 4833, '白': 4824, '为': 4586, '行': 4552, 
      '此': 4535, '见': 4499, '空': 4492, '处': 4474, '在': 4357, '里': 4295, '客': 4295, '清': 4243, '如': 4171, '下': 4149, 
      '是': 4121, '寒': 4116, '得': 4096, '高': 4092, '雨': 4011, '未': 3974, '多': 3922, '明': 3903, '落': 3888, '门': 3841,
       '声': 3821, '青': 3760, '远': 3670, '家': 3612, '路': 3604, '事': 3484, '前': 3457, '应': 3449, '南': 3443, 
    '''
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda  x: -x[1])
    '''
    count_pairs:
    [('，', 123222), ('。', 122489), ('G', 34692), ('E', 34692), ('不', 13522), ('人', 12015), ('山', 10170), ('风', 9680),
     ('日', 8890), ('无', 8586), ('一', 8547), ('云', 7898), ('花', 7436), ('春', 7396), ('来', 7326), ('何', 6874), 
     ('水', 6864), ('月', 6821)]
    '''
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # *count_pairs 可理解为解压
    '''
    *count_pairs:
    ('榸', 1) ('齾', 1) ('兇', 1) ('芚', 1) ('曵', 1) ('熛', 1) ('恞', 1) ('瀔', 1) ('琛', 1) ('俣', 1) ('嗄', 1) ('擉', 1)
     ('扞', 1) ('繙', 1) ('墩', 1) ('輀', 1) ('槭', 1) ('佪', 1) ('刂', 1) ('浟', 1) ('罤', 1) ('菸', 1) ('抉', 1) ('崛', 1) 
     ('咦', 1) ('汐', 1) ('莜', 1) ('歜', 1) ('怎', 1) ('狌', 1) ('呴', 1) ('亳', 1) ('陊', 1) ('垅', 1) ('芫', 1) ('颡', 1) 
     '''
    words, _ = zip(*count_pairs)
    '''
    words
    ('，', '。', 'G', 'E', '不', '人', '山', '风', '日', '无', '一', '云', '花', '春', '来', '何', '水', '月', '上', '有',
     '时', '中', '天', '年', '归', '秋', '相', '知', '长', '去', '自', '君', '心', '夜', '江', '生', '白', '为', '行', 
     '此', '见', '空', '处', '在', '里', '客', '清', '如')
    '''

    # 取前多少个常用字，
    words = words[:len(words)] + (' ',)

    # 每个字映射为一个数字ID
    # zip(words, range(len(words))) 把words按长度打包，然后转为dict
    word_int_map = dict(zip(words, range(len(words))))
    '''
    word_int_map:
    {'，': 0, '。': 1, 'G': 2, 'E': 3, '不': 4, '人': 5, '山': 6, '风': 7, '日': 8, '无': 9, '一': 10, '云': 11, '花': 12, '春': 13, '来': 14}
    '''
    # 把诗集都转成数字的形式
    poems_vector = [list(map(lambda  word: word_int_map.get(word, len(words)), poem)) for poem in poems]
    '''
    poems_vector:
    [
     [2, 34, 67, 634, 0, 2324, 4103, 257, 132, 3321, 1, 4119, 448, 4, 27, 565, 699, 3016, 0, 1625, 14, 250, 49, 820, 32, 2694, 1, 41, 1795, 734, 5, 1355, 1, 3], 
     [2, 482, 133, 258, 1588, 0, 36, 246, 94, 14, 103, 195, 308, 995, 541, 787, 5, 62, 0, 226, 40, 50, 490, 143, 344, 12, 1, 13, 14, 13, 29, 0, 5, 43, 57, 12, 86, 16, 42, 1, 12, 106, 65, 2140, 0, 674, 73, 268, 159, 5, 4, 27, 1, 3]
     ]
    '''
    # 此处用上向量会更好
    # print(poems_vector)
    return poems_vector, word_int_map, words

'''
将诗集按batch_size 分开
'''
def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    # n_chunk 总的数量
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        # 起始和结束位置 从0开始
        start_index = i * batch_size
        end_index = start_index + batch_size

        # 得到了当前的批次
        batches = poems_vec[start_index:end_index]
        # 需要保证所有的输入都是一样的
        # 找到这个batch的所有的poemzhong 最长的poem的长度
        # map() 会根据提供的函数对指定序列做映射。
        # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
        # map(function, iterable, ...)
        length = max(map(len, batches))
        # 填充一个[batch_size, length]规格的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        # print(x_data)
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """

        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches