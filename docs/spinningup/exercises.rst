=========
练习
=========


.. contents:: 目录
    :depth: 2

问题集1：基础实现
---------------------------------------

.. admonition:: 练习 1.1：高斯对数似然

    **练习路径** ``spinup/exercises/problem_set_1/exercise1_1.py``

    **解答路径** ``spinup/exercises/problem_set_1_solutions/exercise1_1_soln.py``

    **指示** 编写一个函数，该函数将Tensorflow符号用于一批对角高斯分布的均值和对数标准，
    以及一个Tensorflow占位符用于从这些分布中生成（先前生成的）样本，
    并返回一个Tensorflow符号以计算那些样品的对数似然。

    你可能会发现复习 `强化学习简介`_ 部分中给出的公式很有用。

    在 ``exercise1_1.py`` 中完成你的实现，并运行该文件以自动检查你的工作。

    **评价标准** 通过使用一批随机输入将输出与已知良好的实现进行比较，来检查你的解决方案。

.. _`强化学习简介`: ../spinningup/rl_intro.html#stochastic-policies


.. admonition:: 练习 1.2：PPO的策略

    **练习路径** ``spinup/exercises/problem_set_1/exercise1_2.py``

    **解答路径** ``spinup/exercises/problem_set_1_solutions/exercise1_2_soln.py``

    **指示** 为PPO实现MLP对角高斯策略。

    在 ``exercise1_2.py`` 中实现你的解决方案，然后运行该文件以自动检查你的工作。

    **评价标准** 你的解决方案将通过在InvertedPendulum-v2 Gym环境中运行20个轮次来进行评估，
    这将花费3-5分钟的时间（取决于你的计算机以及在后台运行的其他进程）。
    成功的标准是最近5个轮次的平均分数超过500，或者最近5个轮次的达到1000（最大可能分数）。

.. admonition:: 练习 1.3：TD3的计算图

    **练习路径** ``spinup/exercises/problem_set_1/exercise1_3.py``

    **解答路径** ``spinup/algos/td3/td3.py``

    **指示** 实现TD3算法的核心计算图。

    作为入门代码，除了计算图外，我们还提供了完整的TD3算法。找到“YOUR CODE HERE”并开始。

    你可能会发现在 `TD3的页面`_ 中查看伪代码很有用。

    在 ``exercise1_3.py`` 中实现你的解决方案，然后运行该文件以查看工作结果。此练习没有自动检查。

    **评价标准** 通过使用HalfCheetah-v2，InvertedPendulum-v2和其他所选的Gym MuJoCo环境
    （通过 ``--env`` 标志设置）运行 ``exercise1_3.py`` 来评估代码。
    它被设置为使用比TD3典型的较小的神经网络（隐藏的大小为[128,128]），最大剧集长度为150，并且仅运行10个轮次。
    目标是要看到相对较快的学习进度（就挂钟时间而言）。实验可能需要约10分钟的时间。

    使用 ``--use_soln`` 标志来运行Spinning Up的TD3，而不是你的实现。
    有趣的是，在10个轮次内，HalfCheetah中的得分应超过300，而InvertedPendulum中的得分应达到150。

.. _`TD3的页面`: ../algorithms/td3.html


问题集2：算法失败模型
--------------------------------------

.. admonition:: 练习 2.1：值函数在TRPO中的拟合

    **练习路径** （不适用，没有代码。）

    **解答路径** `解答在这里 <../spinningup/exercise2_1_soln.html>`_

    许多因素会影响策略梯度算法的性能，但远不及用于优势估计的学习值函数的质量严重。

    在本练习中，你将比较TRPO运行之间的结果，投入了大量精力来拟合值函数（``train_v_iters=80``）
    对比投入很少的精力来拟合值函数（``train_v_iters=0``）。

    **指示** 运行以下命令：

    .. parsed-literal::

        python -m spinup.run trpo --env Hopper-v2 --train_v_iters[v] 0 80 --exp_name ex2-1 --epochs 250 --steps_per_epoch 4000 --seed 0 10 20 --dt

    并绘制结果。（这些实验每个可能需要10分钟左右的时间，而此命令将运行其中的6个。）你发现了什么？

.. admonition:: 练习 2.2：DDPG中的静默错误

    **练习路径** ``spinup/exercises/problem_set_2/exercise2_2.py``

    **解答路径** `解答在这里 <../spinningup/exercise2_2_soln.html>`_

    编写强化学习代码最困难的部分是处理错误，因为故障通常是静默的。
    该代码似乎可以正确运行，但是与无错误的实现相比，该代理的性能将降低，有时甚至永远无法学习任何内容。

    在本练习中，你将观察到一个体内的错误，并将结果与正确的代码进行比较。

    **指示** 运行 ``exercise2_2.py``，这将启动带或不带错误的DDPG实验。
    非调试版本运行DDPG的默认Spinning Up实现，并使用默认方法来创建参与者和评论者网络。
    除了使用错误的方法创建网络之外，错误的版本运行相同的DDPG代码。

    总共将进行六个实验（每种情况下三个随机种子），每个实验都需要10分钟。 完成后，绘制结果。有无bug的性能有何区别？

    在没有引用正确的actor-critic代码的情况下（也就是说，不要在DDPG的 ``core.py`` 文件中查找），请尝试找出错误所在并解释它是如何破坏的。

    **提示** 要找出问题所在，请考虑DDPG代码如何实现DDPG计算图。具体来说，请看以下片段：

    .. code-block:: python

        # Bellman backup for Q function
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

        # DDPG losses
        pi_loss = -tf.reduce_mean(q_pi)
        q_loss = tf.reduce_mean((q-backup)**2)

    actor-critic 代码中的错误如何在这里产生影响？

    **奖励** 是否有任何超参数选择会隐藏该错误的影响？


挑战
----------

.. admonition:: 从头开始编写代码

    正如我们在 `文章 <../spinningup/spinningup.html#learn-by-doing>`_ 中建议的那样，请尝试从头开始重新实现各种深度RL算法。

.. admonition:: 研究要求

    如果你对编写深度学习和强化学习代码深感满意，请考虑尝试在OpenAI的任何常规研究要求上取得进展：

    * `研究要求 1 <https://openai.com/requests-for-research/>`_
    * `研究要求 2 <https://blog.openai.com/requests-for-research-2/>`_
