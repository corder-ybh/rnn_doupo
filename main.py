# 唐诗/歌词项目 入口文件
import argparse

'''
https://puke3615.github.io/2017/10/12/Tensorflow-Poems-Source/
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Intelligence Poem and Lyric Writer')

    help_ = 'you can set this value in terminal --write value can be poem or lyric'
    parser.add_argument('-w', '--write', default='poem', choices=['poem', 'doupo', 'zhetian'], help=help_)

    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)
    parser.set_defaults(train=True)

    args_ = parser.parse_args()
    return args_

#  使用
# python main.py --write poem 训练
if __name__ == '__main__':
    args = parse_args()
    if args.write == 'poem':
        from inference import tang_poems
        if args.train:
            tang_poems.main(True)
        else:
            tang_poems.main(False)
    elif args.write == 'lyric':
        print('lyric')
    elif args.write == 'zhetian':
        from inference import zhetian
        if args.train:
            zhetian.main(True)
        else:
            zhetian.main(False)
    elif args.write == 'doupo':
        from inference import doupo
        if args.train:
            doupo.main(True)
        else:
            doupo.main(False)
    else:
        print("[INFO] write option can only be poem or lyric right now.")