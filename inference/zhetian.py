# 唐诗生成
import collections
import os
import sys
import time
import numpy as np
import tensorflow as tf
# 这里引入可能出错
from models.model import rnn_model
# 句子预处理 产生batch函数
from dataset.fiction import process_poems, generate_batch
import heapq

# 后面那个是说明
tf.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')

# set this to 'main.py' relative path
tf.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/zhetian/'), 'checkpoints save path.')
tf.flags.DEFINE_string('file_path', os.path.abspath('./dataset/data/zhetian.txt'), 'file name of poems.')


tf.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')
tf.flags.DEFINE_string('write', '', 'wtf.')
tf.flags.DEFINE_string('train', '', 'wtf.')
tf.flags.DEFINE_string('no-train', '', 'wtf.')

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

# 起始和结束字符
start_token = 'G'
end_token = 'E'

'''
运行训练，核心
'''
def run_training():
    # 检查点保存路径
    print('its_not_ok:', FLAGS.checkpoints_dir)
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    # 引入预处理
    # 这里返回诗集转换成向量的数据，字与数字映射， 字集
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)

    # batch_size 64 poems_vector 转为数字的映射  word_to_int：字与数字映射
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
    # 返回输入与输出的batch信息

    # 输入、输出 占位符
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), run_size=128, num_layers=2, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    # 保存
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data:batches_inputs[n], output_targets:batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d, batch: %d, training loss: %.6f' % (epoch,batch, loss))

                if epoch % 6 == 0:
                    saver.save(sess, './zhetian_model/', global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now ..')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

def gen_poem(begin_word):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)

    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(vocabularies), run_size=128, num_layers=2,batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint('./zhetian_model/')
        saver.restore(sess, './zhetian_model/-48')

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})

        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''

        while word != end_token:
            print('running')
            poem += word
            x = np.zeros((1, 1))
            x[0,0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']:last_state})
            word = to_word(predict, vocabularies)
        return poem

def pretty_print_poem(poem):
    poem_sentences = poem.split('。 ')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s)

def main(is_train):
    print('zhetian.main:', is_train)
    if is_train:
        print('[INFO] train zhetian fiction...')
        run_training()
    else:
        print('[INFO] write zhetian fiction...')
        begin_word = input('输入起始字:')
        poem2 = gen_poem(begin_word)
        pretty_print_poem(poem2)


if __name__ == '__main__':
    tf.app.run()