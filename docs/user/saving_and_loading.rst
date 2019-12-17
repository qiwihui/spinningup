==================
试验输出
==================

.. contents:: 目录

在本节中，我们将介绍

- Spinning Up算法实现的输出是什么，
- 它们以什么格式存储以及如何组织，
- 它们的存储位置以及如何更改它们，
- 以及如何加载和运行经过训练的策略。

.. admonition:: 你应该知道

    Spinning Up实现目前无法恢复对部分受训练智能体的训练。如果你认为此功能很重要，请告诉我们──或将其视为黑客项目！


算法输出
=================

每种算法都设置为保存训练运行的超参数配置，学习进度，训练过的智能体和值函数，并在可能的情况下保存环境的副本（以便轻松地同时加载智能体和环境）。
输出目录包含以下内容：

.. code::

    +--------------------------------------------------------------------------------+
    | **输出目录结构**                                                                 |
    +----------------+---------------------------------------------------------------+
    |``simple_save/``| | 该目录包含恢复训练的智能体和值函数所需的所有内容。                   |
    |                | | （`详细信息如下`_）                                            |
    +----------------+---------------------------------------------------------------+
    |``config.json`` | | 一个字典，其中包含你用来启动训练功能的args和kwargs的尽可能完整的描述。|
    |                | | 如果你传入了无法序列化为JSON的内容，则日志记录程序应妥善处理它，       |
    |                | | 并且配置文件将使用字符串来表示它。                                 |
    |                | | 注意：这仅用于保存记录。当前不支持从配置文件启动实验。                 |
    +----------------+---------------------------------------------------------------+
    |``progress.txt``| | 制表符分隔的值文件，其中包含日志记录器在整个训练过程中记录的指标记录。  |
    |                | | 比如 ``Epoch``，``AverageEpRet`` 等                            |
    +----------------+---------------------------------------------------------------+
    |``vars.pkl``    | | 包含有关算法状态的任何内容的应保存的 pickle 文件。                 |
    |                | | 当前，所有算法仅使用此方法来保存环境的副本。                       |
    +----------------+---------------------------------------------------------------+

.. admonition:: 你应该知道

    有时，由于无法 pickled 环境而导致环境保存失败，并且 ``vars.pkl`` 为空。
    对于旧版Gym中的Gym Box2D环境，这是一个已知问题，无法以这种方式保存。

.. _`详细信息如下`:

``simple_save`` 目录包含：

.. code::

    +----------------------------------------------------------------------------------+
    | **Simple_Save 文件结构**                                                          |
    +------------------+---------------------------------------------------------------+
    |``variables/``    | | 一个包含Tensorflow Saver的输出的目录。                         |
    |                  | | 请参阅 `Tensorflow SavedModel`_ 的文档。                      |
    +------------------+---------------------------------------------------------------+
    |``model_info.pkl``| | 包含信息（从键到张量名称的映射）的字典，                           |
    |                  | | 可帮助我们在加载后解压缩保存的模型。                              |
    +------------------+---------------------------------------------------------------+
    |``saved_model.pb``| | `Tensorflow SavedModel`_ 所需的 protocol buffer。            |
    +------------------+---------------------------------------------------------------+

.. admonition:: 你应该知道

    您唯一必须“手动”使用的文件是 ``config.json`` 文件。
    我们的智能体程序测试工具将从 ``simple_save/`` 目录和 ``vars.pkl`` 文件中加载内容，
    并且我们的绘图将解释 ``progress.txt`` 的内容，而这些是用于与这些输出接口的正确工具。
    但是没有用于 ``config.json`` 的工具，它只是存在，因此，如果你忘记了要进行实验的超参数，你可以再次检查。

.. _`Tensorflow SavedModel`: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md


保存目录位置
=======================

默认情况下，实验结果将与Spinning Up软件包保存在同一目录下的名为 ``data`` 的文件夹中：

.. parsed-literal::

    spinningup/
        **data/**
            ...
        docs/
            ...
        spinup/
            ...
        LICENSE
        setup.py

你可以通过 ``spinup/user_config.py`` 中的 ``DEFAULT_DATA_DIR`` 修改默认结果目录。

.. _loading-and-running-trained-policies:

加载并运行经过训练的策略
====================================

如果环境成功保存
---------------------------------

对于成功将环境与智能体一起保存的情况，请注意使用以下方法查看训练的智能体在环境中的行为：

.. parsed-literal::

    python -m spinup.run test_policy path/to/output_directory

选项有一些标志：

.. option:: -l L, --len=L, default=0

    *int*. Maximum length of test episode / trajectory / rollout. The default of 0 means no maximum episode length---episodes only end when the agent has reached a terminal state in the environment. (Note: setting L=0 will not prevent Gym envs wrapped by TimeLimit wrappers from ending when they reach their pre-set maximum episode length.)

.. option:: -n N, --episodes=N, default=100

    *int*. Number of test episodes to run the agent for.

.. option:: -nr, --norender

    Do not render the test episodes to the screen. In this case, ``test_policy`` will only print the episode returns and lengths. (Use case: the renderer slows down the testing process, and you just want to get a fast sense of how the agent is performing, so you don't particularly care to watch it.)

.. option:: -i I, --itr=I, default=-1

    *int*. This is an option for a special case which is not supported by algorithms in this package as-shipped, but which they are easily modified to do. Use case: Sometimes it's nice to watch trained agents from many different points in training (eg watch at iteration 50, 100, 150, etc.). The logger can do this---save snapshots of the agent from those different points, so they can be run and watched later. In this case, you use this flag to specify which iteration to run. But again: spinup algorithms by default only save snapshots of the most recent agent, overwriting the old snapshots. 

    The default value of this flag means "use the latest snapshot."

    To modify an algo so it does produce multiple snapshots, find the following lines (which are present in all of the algorithms):

    .. code-block:: python

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

    and tweak them to

    .. code-block:: python

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)

    Make sure to then also set ``save_freq`` to something reasonable (because if it defaults to 1, for instance, you'll flood your output directory with one ``simple_save`` folder for each snapshot---which adds up fast).


.. option:: -d, --deterministic

    Another special case, which is only used for SAC. The Spinning Up SAC implementation trains a stochastic policy, but is evaluated using the deterministic *mean* of the action distribution. ``test_policy`` will default to using the stochastic policy trained by SAC, but you should set the deterministic flag to watch the deterministic mean policy (the correct evaluation policy for SAC). This flag is not used for any other algorithms.


找不到环境错误
---------------------------

如果未成功保存环境，则可能导致 ``test_policy.py`` 崩溃

.. parsed-literal::

    Traceback (most recent call last):
      File "spinup/utils/test_policy.py", line 88, in <module>
        run_policy(env, get_action, args.len, args.episodes, not(args.norender))
      File "spinup/utils/test_policy.py", line 50, in run_policy
        "page on Experiment Outputs for how to handle this situation."
    AssertionError: Environment not found!

     It looks like the environment wasn't saved, and we can't run the agent in it. :( 

     Check out the readthedocs page on Experiment Outputs for how to handle this situation.

在这种情况下，只要你可以地重新创建环境，就可以观察智能体的执行情况。在IPython中尝试以下操作：

>>> from spinup.utils.test_policy import load_policy, run_policy
>>> import your_env
>>> _, get_action = load_policy('/path/to/output_directory')
>>> env = your_env.make()
>>> run_policy(env, get_action)
Logging data to /tmp/experiments/1536150702/progress.txt
Episode 0    EpRet -163.830      EpLen 93
Episode 1    EpRet -346.164      EpLen 99
...


使用经过训练的值函数
-----------------------------

``test_policy.py`` 工具无法帮助你查看经过训练的值函数，如果要使用这些函数，则必须手工进行一些挖掘。
有关详细信息，请查看 `restore_tf_graph`_ 函数的文档。

.. _`restore_tf_graph`: ../utils/logger.html#spinup.utils.logx.restore_tf_graph
