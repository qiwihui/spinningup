============
安装
============

.. contents:: 目录

Spinning Up 要求 Python3, OpenAI Gym, and OpenMPI.

Spinning Up 现在只支持 Linux 和 OSX。尽管尚未经过广泛测试，但可以在Windows上安装。[#]_

.. admonition:: 你应该知道

    Spinning Up中的许多示例和基准都是针对使用 `MuJoCo`_ 物理引擎的RL环境。MuJoCo是需要许可证的专有软件，该许可证可免费试用，对学生免费，其它则收费。
    安装它是可选的，但是由于它对研究界很重要，它是在连续控制中对深度强化学习算法进行基准测试的事实上的标准，因此，建议安装。

    不过，如果你决定不安装MuJoCo，请不要担心。你绝对可以通过在Gym的 `Classic Control`_ 和 `Box2d`_ 环境上运行算法来完全开始学习强化学习，这是完全免费的。

.. [#] 似乎至少有一个人找出了一种 `在Windows上运行的解决方法`_。 如果您尝试其他方法并成功，请告诉我们你的做法！

.. _`Classic Control`: https://gym.openai.com/envs/#classic_control
.. _`Box2d`: https://gym.openai.com/envs/#box2d
.. _`MuJoCo`: http://www.mujoco.org/index.html
.. _`在Windows上运行的解决方法`: https://github.com/openai/spinningup/issues/23


安装 Python
=================

我们建议通过 Anaconda 安装 Python。
Anaconda是一个包含Python和许多有用的Python软件包的库，以及一个名为conda的环境管理器，它使软件包管理变得简单。

此处按照Anaconda的 `安装说明`_ 进行操作。下载并安装Anaconda3（撰写本文时，`Anaconda3-5.3.0`_）。
然后创建一个conda Python 3.6 环境来组织Spinning Up中使用的软件包：

.. parsed-literal::

    conda create -n spinningup python=3.6

要在刚创建的环境中使用Python，请使用以下方法激活环境：

.. parsed-literal::

    conda activate spinningup

.. admonition:: 你应该知道

    如果你是python环境和软件包管理的新手，那么这些内容可能会很快变得令人困惑或不知所措，并且一路上你可能会遇到一些麻烦。
    （特别是，你应该会遇到诸如“我刚刚安装了此东西，但是它说它在我尝试使用时找不到的问题！”之类的问题。）
    你可能需要通读一些干净的解释，有关什么是软件包管理，为什么这样做是个好主意以及通常必须执行哪些命令才能正确使用它。

    `FreeCodeCamp`_ 有一个很好的解释，值得一读。 在 `Towards Data Science`_ 上有一个简短的描述，该描述也很有帮助且内容丰富。
    最后，如果你是一个非常有耐心的人，则可能需要阅读 `Conda的文档页面`_ （枯燥但非常有用的）。

.. _`安装说明`: https://docs.continuum.io/anaconda/install/
.. _`Anaconda3-5.3.0`: https://repo.anaconda.com/archive/
.. _`FreeCodeCamp`: https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c
.. _`Towards Data Science`: https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097
.. _`Conda的文档页面`: https://conda.io/docs/user-guide/tasks/manage-environments.html
.. _`this Github issue for Tensorflow`: https://github.com/tensorflow/tensorflow/issues/20444


安装 OpenMPI
==================

Ubuntu
------

.. parsed-literal::

    sudo apt-get update && sudo apt-get install libopenmpi-dev


Mac OS X
--------

在Mac上安装系统软件包需要 Homebrew_。 安装Homebrew后，运行以下命令：

.. parsed-literal::

    brew install openmpi

.. _Homebrew: https://brew.sh


安装 Spinning Up
======================

.. parsed-literal::

    git clone https://github.com/openai/spinningup.git
    cd spinningup
    pip install -e .

.. admonition:: 你应该知道

    Spinning Up默认情况下会安装Gym的所有内容，**除了** MuJoCo环境。如果你在安装Gym时遇到任何麻烦，请查看`Gym`_ github页面以获取帮助。
    如果要使用MuJoCo环境，请参见下面的可选安装部分。

.. _`Gym`: https://github.com/openai/gym


检查你的安装
==================

要查看是否已成功安装Spinning Up，请尝试在 LunarLander-v2 环境中使用以下命令运行PPO：

.. parsed-literal::

    python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999

该过程可能会持续10分钟左右，你可以在继续阅读文档的同时将其保留在后台。
这不会训练智能完成任务，但是会运行足够长的时间，以至于结果出现时你可以看到 *一些* 学习进度。

训练结束后，观看有关训练过的策略的视频

.. parsed-literal::

    python -m spinup.run test_policy data/installtest/installtest_s0

并绘制结果

.. parsed-literal::

    python -m spinup.run plot data/installtest/installtest_s0


安装 MuJoCo （可选）
============================

首先，转到 `mujoco-py`_ github页面。请遵循 README 文件中的安装说明，
该说明描述了如何安装MuJoCo物理引擎和mujoco-py软件包（允许在Python中使用MuJoCo）。

.. admonition:: 你应该知道

    为了使用MuJoCo仿真器，您将需要获得`MuJoCo许可证`_。任何人均可享受30天免费许可证，全日制学生可享受1年免费许可证。

安装MuJoCo后，请使用以下命令安装相应的Gym环境

.. parsed-literal::

    pip install gym[mujoco,robotics]

然后通过在Walker2d-v2环境中运行PPO来检查一切是否正常

.. parsed-literal::

    python -m spinup.run ppo --hid "[32,32]" --env Walker2d-v2 --exp_name mujocotest


.. _`mujoco-py`: https://github.com/openai/mujoco-py
.. _`MuJoCo许可证`: https://www.roboti.us/license.html
