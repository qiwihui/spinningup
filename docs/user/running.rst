===================
运行试验
===================


.. contents:: 目录

体验深度强化的最佳方法之一是运行算法，并查看它们在不同任务上的执行情况。
Spinning Up代码库使小规模（本地）实验更容易实现，在本节中，我们将讨论两种运行方式：从命令行运行，或者通过脚本中的函数调用。


从命令行启动
===============================

Spinning Up提供了 ``spinup/run.py``，这是一个方便的工具，可让你从命令行轻松启动任何算法（可以选择任何超参数）。
它也充当工具的精简包装，用于观看训练过的策略和绘图，尽管我们不会在此页上讨论该功能
（有关这些详细信息，请参见 `试验输出`_ 和 `绘制结果`_）。

从命令行运行Spinning Up算法的标准方法是

.. parsed-literal::

    python -m spinup.run [algo name] [experiment flags]

例如：

.. parsed-literal::

    python -m spinup.run ppo --env Walker2d-v2 --exp_name walker

.. _`试验输出`: ../user/saving_and_loading.html
.. _`绘制结果`: ../user/plotting.html

.. admonition:: 你应该知道

    如果使用ZShell：ZShell会将方括号解释为特殊字符。Spinning Up在命令行参数中以几种方式使用方括号。
    请确保对其进行转义，或者如果你希望默认情况下对其进行转义，请尝试 `此处`_ 推荐的解决方案。

.. _`此处`: http://kinopyo.com/en/blog/escape-square-bracket-by-default-in-zsh

.. admonition:: 详细快速开始指南

    .. parsed-literal::

        python -m spinup.run ppo --exp_name ppo_ant --env Ant-v2 --clip_ratio 0.1 0.2
            --hid[h] [32,32] [64,32] --act tf.nn.tanh --seed 0 10 20 --dt
            --data_dir path/to/data

    在 ``Ant-v2`` Gym 环境中运行 PPO，可以使用多种配置。

    ``clip_ratio``， ``hid`` 和 ``act`` 用于设置算法超参数。你可以为超参数提供多个值以运行多个实验。
    检查文档以查看可以设置的超参数（单击此处获取 `PPO文档`_）。

    ``hid`` 和 ``act`` 是 `特殊快捷标志`_ 用于为算法训练的神经网络设置隐藏层大小和激活函数。

    ``seed`` 标志设置随机数生成器的种子。强化学习算法具有较高的方差，因此请尝试多个种子以了解性能如何变化。

    ``dt`` 标志可确保保存目录名称中包含时间戳
    （否则，则不会包含时间戳，除非你在 ``spinup/user_config.py`` 的``FORCE_DATESTAMP=True`` 设置）。

    ``data_dir`` 标志允许你设置保存结果的路径。
    默认值由 ``spinup/user_config.py`` 中的 ``DEFAULT_DATA_DIR`` 设置，
    它将是 ``spinningup`` 文件夹中的 ``data`` 子文件夹（除非你进行更改）。

    `保存文件夹名称`_ 基于 ``exp_name`` 和具有多个值的所有标志。在目录名称中会出现一个简写，而不是完整的标志。
    用户可以在标志后的方括号中提供简写方式，例如 ``--hid[h]``。否则，快捷是标志的子字符串（``clip_ratio`` 变为 ``cli``）。
    为了说明这一点，以 ``clip_ratio=0.1``，``hid=[32,32]`` 和 ``seed=10`` 运行的保存目录将是：

    .. parsed-literal::

        path/to/data/YY-MM-DD_ppo_ant_cli0-1_h32-32/YY-MM-DD_HH-MM-SS-ppo_ant_cli0-1_h32-32_seed10

.. _`PPO文档`: ../algorithms/ppo.html#spinup.ppo
.. _`特殊快捷标志`: ../user/running.html#shortcut-flags
.. _`保存文件夹名称`: ../user/running.html#where-results-are-saved


从命令行设置超参数
---------------------------------------------

每种算法中的每个超参数都可以直接从命令行进行控制。
如果 ``kwarg`` 是算法函数调用的有效关键字参数，则可以使用标志 ``--kwarg`` 为其设置值。
要找出可用的关键字参数，请参见文档页面中的算法，或尝试

.. parsed-literal::

    python -m spinup.run [algo name] --help

查看文档字符串的输出。

.. admonition:: 你应该知道

    值在使用前先通过 ``eval()``，因此你可以直接从命令行描述一些函数和对象。例如：

    .. parsed-literal::

        python -m spinup.run ppo --env Walker2d-v2 --exp_name walker --act tf.nn.elu

    设置 ``tf.nn.elu`` 为激活函数。

.. admonition:: 你应该知道

    对于采用dict值的参数，有一些不错的处理方法。无需提供

    .. parsed-literal::

        --key dict(v1=value_1, v2=value_2)

    你可以使用

    .. parsed-literal::

        --key:v1 value_1 --key:v2 value_2

    来获得同样的结果。


一次启动多个实验
--------------------------------------

您可以通过简单地为给定参数提供多个值来启动要 **串联** 执行的多个实验。（将针对每种可能的值组合进行实验。）

例如，要启动具有不同随机种子（0、10和20）的等效运行，请执行以下操作：

.. parsed-literal::

    python -m spinup.run ppo --env Walker2d-v2 --exp_name walker --seed 0 10 20

实验无法并行启动，因为它们会占用足够的资源，因此无法同时执行多个实验，因此无法加快速度。


特殊标志
-------------

一些标志受到特殊对待。


环境标志
^^^^^^^^^^^^^^^^

.. option:: --env, --env_name

    *string*. The name of an environment in the OpenAI Gym. All Spinning Up algorithms are implemented as functions that accept ``env_fn`` as an argument, where ``env_fn`` must be a callable function that builds a copy of the RL environment. Since the most common use case is Gym environments, though, all of which are built through ``gym.make(env_name)``, we allow you to just specify ``env_name`` (or ``env`` for short) at the command line, which gets converted to a lambda-function that builds the correct gym environment.

.. _shortcut-flags:

快捷标志
^^^^^^^^^^^^^^

一些算法参数相对较长，我们为它们启用了快捷方式：

.. option:: --hid, --ac_kwargs:hidden_sizes

    *list of ints*. Sets the sizes of the hidden layers in the neural networks (policies and value functions).

.. option:: --act, --ac_kwargs:activation

    *tf op*. The activation function for the neural networks in the actor and critic.

这些标志对于所有当前的Spinning Up算法均有效。

配置标志
^^^^^^^^^^^^

这些标志不是任何算法的超参数，而是以某种方式更改实验配置。

.. option:: --cpu, --num_cpu

    *int*. If this flag is set, the experiment is launched with this many processes, one per cpu, connected by MPI. Some algorithms are amenable to this sort of parallelization but not all. An error will be raised if you try setting ``num_cpu`` > 1 for an incompatible algorithm. You can also set ``--num_cpu auto``, which will automatically use as many CPUs as are available on the machine.

.. option:: --exp_name

    *string*. The experiment name. This is used in naming the save directory for each experiment. The default is "cmd" + [algo name].

.. option:: --data_dir

    *path*. Set the base save directory for this experiment or set of experiments. If none is given, the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py`` will be used.

.. option:: --datestamp

    *bool*. Include date and time in the name for the save directory of the experiment.


.. _where-results-are-saved:

保存结果的位置
-----------------------

特定实验的结果（单次运行的超参数配置）存储在

::

    data_dir/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]_s[seed]

其中

* ``data_dir`` 是标志 ``--data_dir`` 的值（如果 ``--data_dir`` 没有设置，
  默认为 ``spinup/user_config.py`` 中的 ``DEFAULT_DATA_DIR``），
* 如果设置了 ``--datestamp`` 标志， ``outer_prefix`` 是 ``YY-MM-DD_`` 格式的时间戳，否则为空，
* 如果设置了 ``--datestamp`` 标志，``inner_prefix`` 是 ``YY-MM-DD_HH-MM-SS-`` 格式的时间戳，
  否则为空，
* ``suffix`` 是基于实验超参数的特殊字符串。

后缀如何确定？
^^^^^^^^^^^^^^^^^^^^^^^^^

仅当您一次运行多个实验时才包含后缀，并且后缀仅包含对跨实验而不同的超参数的引用，随机种子除外。
目的是确保相似实验的结果（共享除种子外的所有参数的实验）被分组在同一文件夹中。

后缀是通过将超参数的 **简写** 及其值组合在一起来构造的，其中简写可以是1）根据超参数名称自动构建，也可以是2）用户提供。
用户可以通过在kwarg标志后的方括号中书写来提供速记。

例如，考虑：

.. parsed-literal::

    python -m spinup.run ddpg --env Hopper-v2 --hid[h] [300] [128,128] --act tf.nn.tanh tf.nn.relu

在此，``--hid`` 标志具有 **用户提供的简写** ``h``。 用户未提供 ``--act`` 标志的简写，因此将自动为其构造一个标志。

在这种情况下产生的后缀是：

.. parsed-literal::
    _h128-128_ac-actrelu
    _h128-128_ac-acttanh
    _h300_ac-actrelu
    _h300_ac-acttanh

注意，``h`` 是由用户给定的。``ac-act`` 简写由 ``ac_kwargs:activation``（``act`` 标志的真实名称）构造而成。


其他
-----

.. admonition:: 你实际上不需要知道这一点

    每个单独的算法都位于文件 ``spinup/algos/ALGO_NAME/ALGO_NAME.py`` 中，
    并且这些文件可以使用有限的一组参数直接从命令行运行（其中一些参数与 ``spinup/run.py`` 的可用参数不同）。
    各个算法文件中的命令行支持基本上是残留的，但是，这 **不是** 执行实验的推荐方法。

    本文档页面将不描述这些命令行调用，而 *仅* 描述通过 ``spinup/run.py`` 进行的调用。


从脚本启动
======================

每种算法都实现为python函数，可以直接从 ``spinup`` 包中导入，例如

>>> from spinup import ppo

有关每种算法的完整说明，请参见文档页面。 这些方法可用于建立专门的自定义实验，例如：

.. code-block:: python

    from spinup import ppo
    import tensorflow as tf
    import gym

    env_fn = lambda : gym.make('LunarLander-v2')

    ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

    logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)


使用ExperimentGrid
--------------------

在机器学习研究中运行具有许多可能的超参数的相同算法通常很有用。
Spinning Up附带了一个用于简化此过程的简单工具，称为 `ExperimentGrid`_。

考虑 ``spinup/examples/bench_ppo_cartpole.py`` 中的例子：

.. code-block:: python
   :linenos:

    from spinup.utils.run_utils import ExperimentGrid
    from spinup import ppo
    import tensorflow as tf

    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cpu', type=int, default=4)
        parser.add_argument('--num_runs', type=int, default=3)
        args = parser.parse_args()

        eg = ExperimentGrid(name='ppo-bench')
        eg.add('env_name', 'CartPole-v0', '', True)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('epochs', 10)
        eg.add('steps_per_epoch', 4000)
        eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
        eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
        eg.run(ppo, num_cpu=args.cpu)

创建了ExperimentGrid对象后，将参数添加到其中

.. parsed-literal::

    eg.add(param_name, values, shorthand, in_name)

其中，``in_name`` 会强制参数显示在实验名称中，即使该参数在所有实验中都具有相同的值。

添加所有参数后，

.. parsed-literal::

    eg.run(thunk, **run_kwargs)

通过将配置作为函数 ``thunk`` 的kwarg提供，在网格中运行所有实验（每个有效配置对应一个实验）。
``ExperimentGrid.run`` 使用名为 `call_experiment`_ 的函数来启动 ``thunk``，
``**run_kwargs`` 指定 ``call_experiment`` 的行为。有关详细信息，请参见 `文档页面`_。

除了没有快捷键kwargs（在 ``ExperimentGrid`` 中不能对 ``ac_kwargs:hidden_sizes`` 使用 ``hid``）
之外，``ExperimentGrid`` 的基本行为与从命令行运行事物相同。
（实际上，``spinup.run`` 在后台使用了 ``ExperimentGrid``。）

.. _`ExperimentGrid`: ../utils/run_utils.html#experimentgrid
.. _`文档页面`: ../utils/run_utils.html#experimentgrid
.. _`call_experiment`: ../utils/run_utils.html#spinup.utils.run_utils.call_experiment
