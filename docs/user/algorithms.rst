==========
算法
==========

.. contents:: 目录

包括哪些算法
===============

下面的算法已经在 Spinning Up 包中实现:

- `Vanilla Policy Gradient`_ (VPG)
- `Trust Region Policy Optimization`_ (TRPO)
- `Proximal Policy Optimization`_ (PPO)
- `Deep Deterministic Policy Gradient`_ (DDPG)
- `Twin Delayed DDPG`_ (TD3)
- `Soft Actor-Critic`_ (SAC)

这些算法全部以 `多层感知机`_ （非递归）actor-critics 的方式实现，
从而适用于全观察、基于非图像的强化学习环境，例如 `Gym Mujoco`_ 环境。

.. _`Gym Mujoco`: https://gym.openai.com/envs/#mujoco
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html
.. _`Trust Region Policy Optimization`: ../algorithms/trpo.html
.. _`Proximal Policy Optimization`: ../algorithms/ppo.html
.. _`Deep Deterministic Policy Gradient`: ../algorithms/ddpg.html
.. _`Twin Delayed DDPG`: ../algorithms/td3.html
.. _`Soft Actor-Critic`: ../algorithms/sac.html
.. _`多层感知机`: https://en.wikipedia.org/wiki/Multilayer_perceptron


为什么使用这些算法？
=====================

我们在这个项目中选取了能够呈现强化学习近些年思想发展的核心深度强化学习算法。
特别是两种算法 PPO 和 SAC，在策略学习算法中，它们在可靠性采样效率方面都非常先进。
它们还揭示了在深度强化学习中设计和使用算法时要做出的一些权衡。

同轨策略（On-Policy）算法
--------------------------

Vanilla Policy Gradient（VPG） 是深度强化学习领域最基础也是入门级的算法，发表时间远早于深度强化学习。
VPG 算法的核心思想可以追溯到上世纪80年代末90年代初。在那之后，TRPO和PPO等更强大的算法才相继诞生。

上述系列工作都是基于不使用历史数据的 *同轨策略*，也就是说，它们不使用旧数据，因此在采样效率上表现相对较差。
但这也是有原因的：这些算法直接优化我们关心的目标──策略表现，并且从数学上计算出需要同轨策略数据来计算更新。
因此，这一系列算法在权衡采样效率的同时，还考虑了稳定性，但你会看到技术的进步（从VPG到TRPO到PPO）弥补了采样效率的不足。

异轨策略（Off-Policy）算法
---------------------------

DDPG是类似于VPG的基础算法，尽管它的提出时间较晚，导致DDPG产生的确定性策略梯度理论直到2014年才发布。
DDPG与Q-learning算法紧密相关，都是同时学习Q函数和策略并通过更新相互改进。

诸如DDPG和Q-Learning之类的算法是 *异轨策略*，因此它们能够非常有效地重用旧数据。
他们通过利用贝尔曼方程获得最优性而获得了这一好处，Q函数可以通过训练满足使用 *任何* 环境交互数据（只要有来自环境中高回报区域的足够经验）。

但问题是，满足贝尔曼方程并不能保证一定有很好的策略性能。
*根据经验*，满足贝尔曼方程可以有不错的性能以及很好的采样效率，但是缺少保证会使此类算法变得脆弱而不稳定。
基于DDPG的后续工作 TD3 和 SAC 提出了很多新的方案来缓解这些问题。


代码格式
===========

Spinning Up 项目的算法都按照标准的模板来实现。每个算法由两个文件组成：算法文件，主要是算法的核心逻辑，以及核心文件，包括各种运行算法所需的工具类。

算法文件
------------------

算法文件始终以经验缓存对象的类定义开头，该类定义用于存储智能体与环境之间的交互信息。

接下来有一个函数可以运行算法，并执行以下任务（按此顺序）：

    1) Logger 设定

    2) 随机种子的设定

    3) 环境实例化

    4) 为计算图创建 placeholder

    5) 通过 ``actor_critic`` 函数传递算法函数作为参数构建actor-critic计算图

    6) 实例化经验缓存

    7) 为算法特定的损失函数和诊断建立计算图

    8) 配置训练参数

    9) 构建 TF Session 并初始化参数

    10) 通过 logger 设置模型保存

    11) 定义运行算法主循环需要的函数（例如核心更新函数，获取动作函数，测试智能体函数等，取决于具体的算法）

    12) 运行算法主循环

        a) 让智能体在环境中开始运行

        b) 根据算法的主要方程式，周期性更新参数

        c) 记录核心性能指标并保存智能体

最后，是可以从命令行在Gym环境中直接运行该算法的支持。

核心文件
-------------

核心文件并没有像算法文件那样严格遵守模板，但也有一些相似的结构。

    1) 构建和管理 placeholder 的函数

    2) 用于构建与特定算法的 ``actor_critic`` 方法相关的计算图部分的函数

    3) 其他有用的函数

    4) 与算法兼容的多层感知机 actor-critic 实现，策略和值函数都是通过简单的多层感知机来表示
