=========
日志记录
=========

.. contents:: Table of Contents

使用Logger
==============

Spinning Up 提供了基本的日志工具，在 `Logger`_ 和 `EpochLogger`_ 类中实现。
Logger类包含用于保存诊断，超参数配置，训练运行的状态和训练好的模型。
EpochLogger类在其之上添加了一层，以便于在每个轮次和整个MPI工作者之间轻松跟踪诊断的平均值，标准差，最小值和最大值。

.. admonition:: 你应该知道

    所有 Spinning Up 算法实现使用了 EpochLogger。

.. _`Logger`: ../utils/logger.html#spinup.utils.logx.Logger
.. _`EpochLogger`: ../utils/logger.html#spinup.utils.logx.EpochLogger


例子
--------

首先，让我们看一个简单的示例，说明EpochLogger如何跟踪诊断值：

>>> from spinup.utils.logx import EpochLogger
>>> epoch_logger = EpochLogger()
>>> for i in range(10):
        epoch_logger.store(Test=i)
>>> epoch_logger.log_tabular('Test', with_min_and_max=True)
>>> epoch_logger.dump_tabular()
-------------------------------------
|     AverageTest |             4.5 |
|         StdTest |            2.87 |
|         MaxTest |               9 |
|         MinTest |               0 |
-------------------------------------

``store`` 方法用于将所有 ``Test`` 值保存到 ``epoch_logger`` 的内部状态。
然后，在调用 ``log_tabular`` 时，它将计算内部状态下所有值的 ``Test`` 的平均值，标准偏差，最小值和最大值。
调用 ``log_tabular`` 之后，内部状态会清除干净（以防止在下一个轮次泄漏到统计信息中）。
最后，调用 ``dump_tabular`` 将诊断信息写入文件和标准输出。

接下来，让我们看一下包含日志记录的完整训练过程，以突出显示配置和模型保存以及诊断记录：

.. code-block:: python
   :linenos:
   :emphasize-lines: 18, 19, 42, 43, 54, 58, 61, 62, 63, 64, 65, 66

    import numpy as np
    import tensorflow as tf
    import time
    from spinup.utils.logx import EpochLogger


    def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


    # Simple script for training an MLP on MNIST.
    def train_mnist(steps_per_epoch=100, epochs=5, 
                    lr=1e-3, layers=2, hidden_size=64, 
                    logger_kwargs=dict(), save_freq=1):

        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Load and preprocess MNIST data
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28*28) / 255.0

        # Define inputs & main outputs from computation graph
        x_ph = tf.placeholder(tf.float32, shape=(None, 28*28))
        y_ph = tf.placeholder(tf.int32, shape=(None,))
        logits = mlp(x_ph, hidden_sizes=[hidden_size]*layers + [10], activation=tf.nn.relu)
        predict = tf.argmax(logits, axis=1, output_type=tf.int32)

        # Define loss function, accuracy, and training op
        y = tf.one_hot(y_ph, 10)
        loss = tf.losses.softmax_cross_entropy(y, logits)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_ph, predict), tf.float32))
        train_op = tf.train.AdamOptimizer().minimize(loss)

        # Prepare session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_ph}, 
                                    outputs={'logits': logits, 'predict': predict})

        start_time = time.time()

        # Run main training loop
        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                idxs = np.random.randint(0, len(x_train), 32)
                feed_dict = {x_ph: x_train[idxs],
                             y_ph: y_train[idxs]}
                outs = sess.run([loss, acc, train_op], feed_dict=feed_dict)
                logger.store(Loss=outs[0], Acc=outs[1])

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state(state_dict=dict(), itr=None)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Acc', with_min_and_max=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('TotalGradientSteps', (epoch+1)*steps_per_epoch)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    if __name__ == '__main__':
        train_mnist()

In this example, observe that

* 第19行，`logger.save_config`_ 用来将超参数配置保存到JSON文件中。
* 第42和43行，`logger.setup_tf_saver`_ 用于准备日志记录以保存计算图的关键元素。
* 第54行，通过`logger.store`_ 将诊断保存到日志记录的内部状态。
* 第58行，计算图每个轮次通过 `logger.save_state`_ 保存一次。
* 第61-66行，`logger.log_tabular`_ 和 `logger.dump_tabular`_ 用于将轮次诊断写入文件。
  请注意，传递到 `logger.log_tabular`_ 的键与传递到 `logger.store`_ 的键相同。

.. _`logger.save_config`: ../utils/logger.html#spinup.utils.logx.Logger.save_config
.. _`logger.setup_tf_saver`: ../utils/logger.html#spinup.utils.logx.Logger.setup_tf_saver
.. _`logger.store`: ../utils/logger.html#spinup.utils.logx.EpochLogger.store
.. _`logger.save_state`: ../utils/logger.html#spinup.utils.logx.Logger.save_state
.. _`logger.log_tabular`: ../utils/logger.html#spinup.utils.logx.EpochLogger.log_tabular
.. _`logger.dump_tabular`: ../utils/logger.html#spinup.utils.logx.Logger.dump_tabular


日志记录和MPI
---------------

.. admonition:: 你应该知道

    通过使用MPI求平均梯度和/或其他关键数量，可以轻松地并行化强化学习中的几种算法。
    Spinning Up日志记录的设计使其在使用MPI时表现良好：只会从rank 0的进程中写入标准输出。
    但是，如果你使用EpochLogger，其他进程的信息也不会丢失：通过 ``store`` 传递到EpochLogger中的数据，
    无论存储在哪个进程中，都将用于计算诊断的平均值/标准差/最小值/最大值。


Logger类
==============

.. autoclass:: spinup.utils.logx.Logger
    :members:

    .. automethod:: spinup.utils.logx.Logger.__init__

.. autoclass:: spinup.utils.logx.EpochLogger
    :show-inheritance:
    :members:


加载保存的图
====================

.. autofunction:: spinup.utils.logx.restore_tf_graph

当你使用此方法还原由Spinning Up实现保存的图时，可以最少期望它包括以下内容：

======  ===============================================
键      值
======  ===============================================
``x``   Tensorflow 状态输入占位符。
``pi``  以 ``x`` 中的状态为条件，从智能体中采样动作。
======  ===============================================

通常还存储算法的相关值函数。 有关给定算法还能保存哪些内容的详细信息，请参见其文档页面。
