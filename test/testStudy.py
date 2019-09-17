# coding=utf-8
import argparse
from tensorflow.python.client import device_lib

# # 创建一个解析器
# parser = argparse.ArgumentParser(description='Process some integers.')
#
# # 添加参数 ArgumentParser 添加程序参数信息通过add_argument()方法完成。通常，这些调用指定ArgumentParser如何获取命令行字符串并将其转换为对象
# parser.add_argument('integers', metavar='N', type=int, help='an interger for the accumulator')
#
# args = parser.parse_args()
# print(args.accumulate(args.integers))

parser = argparse.ArgumentParser(description='Process some integers.')
'''
name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。 选项会以 - 前缀识别，剩下的参数则会被假定为位置参数:
action - 当参数在命令行中出现时使用的动作基本类型。
nargs - 命令行参数应当消耗的数目。
const - 被一些 action 和 nargs 选择所需求的常数。
default - 当参数未在命令行中出现时使用的值。
type - 命令行参数应当被转换成的类型。
choices - 可用的参数的容器。
required - 此命令行选项是否可省略 （仅选项可用）。
help - 一个此选项作用的简单描述。
metavar - 在使用方法消息中使用的参数值示例。
dest - 被添加到 parse_args() 所返回对象上的属性名。
'''
'''
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.integers)
'''

def parse_args():
    # 说明内容
    parser = argparse.ArgumentParser(description='Intelligence Poem and Lyric Writer')

    help_ = 'you can set this value in terminal --write value can be poem or lyric'
    parser.add_argument('-w', '--write', default='poem', choices=['poem', 'lyric'], help=help_)

    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)
    parser.set_defaults(train=True)

    args_ = parser.parse_args()
    return args_

if __name__ == '__main__':
    args = parse_args()
    if args.write == 'poem':
        print('poem...')
        if args.train:
            print("train...")
            print(device_lib.list_local_devices())
        else:
            print("no train...")
    elif args.write == 'lyric':
        print('lyric...')
    else:
        print("[INFO] write option can only be poem or lyric right now.")