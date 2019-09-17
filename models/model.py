import tensorflow as tf
import numpy as np

'''
跑模型
'''
def rnn_model(model, input_data, output_data, vocab_size, run_size=128,
              num_layers=2, batch_size=64, learning_rate=0.01):
    end_points = {}

    '''
    GRUCell 与 BasicLSTMCell 的区别不清楚
    但是BasicLSTMCell比BasicRNNCell高级
    '''
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell
    # 构造具体的cell
    cell = cell_fun(run_size, state_is_tuple=True)
    # 将单层的cell变为更深的cell, 以表征更复杂的关联关系
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    # 初始化cell的状态
    if output_data is not None:
        # 训练时batch容量为batch_size
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        # 使用时batch容量为1
        initial_state = cell.zero_state(1, tf.float32)

    # tensorflow对于lookup_embedding的操作只能在cpu上进行
    '''
    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 8542185094111138873
]
    '''
    with tf.device("/cpu:0"):
        # 构造(vocab_size + 1, run_size)的Tensor
        # len(vocabularies)
        # embedding_lookup函数
        # output = embedding_lookup(embedding, ids): 将ids里的element替换为embedding中对应element位的值
        # 即: embedding: [[1, 2], [3, 4], [5, 6]]  ids: [1, 2]  则outputs: [[3, 4], [5, 6]]
        # 类比one_hot, 只是这里是x_hot
        # embedding: (3, 2)  ids: (10, )  outputs: (10, 2)
        # 处理之后的shape为(batch_size, n_steps, rnn_size)
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, run_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, run_size])

    weights = tf.Variable(tf.truncated_normal([run_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state

    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction
    return end_points