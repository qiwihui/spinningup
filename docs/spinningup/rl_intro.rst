================================
第一部分：强化学习中的核心概念
================================


.. contents:: 目录
    :depth: 2

欢迎来到强化学习的介绍部分！我们希望你能了解以下内容：

* 用于讨论该主题的语言和符号，
* 高层次的理解：关于强化学习算法做什么（我们会尽量避免 *如何做* 这个话题），
* 少量算法背后的核心数学知识。

总的来说，强化学习是关于智能体以及它们如何通过试错来学习的研究。
它确定了通过奖励或惩罚智能体的动作从而使它未来更容易重复或者放弃某一动作的思想。


强化学习能做什么？
===================

强化学习的方法最近已经在很多地方取得了成功。例如，它被用来教电脑在仿真环境下控制机器人：

.. raw:: html

    <video autoplay="" src="https://d4mucfpksywv.cloudfront.net/openai-baselines-ppo/knocked-over-stand-up.mp4" loop="" controls="" style="display: block; margin-left: auto; margin-right: auto; margin-bottom:1.5em; width: 100%; max-width: 720px; max-height: 80vh;">
    </video>

以及在现实世界中：

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/jwSbzNHGflM?ecver=1" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>
    <br />

强化学习因为被用在复杂策略游戏创造出突破性的 AI 中而名声大噪，
最著名的要数 `围棋`_ 、`Dota`_、教计算机从原始像素 `玩Atari游戏`_ 以及训练模拟机器人 `听从人类的指令`_。

.. _`围棋`: https://deepmind.com/research/alphago/
.. _`Dota`: https://blog.openai.com/openai-five/
.. _`玩Atari游戏`: https://deepmind.com/research/dqn/
.. _`听从人类的指令`: https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/


核心概念和术语
============================

.. figure:: ../images/rl_diagram_transparent_bg.png
    :align: center

    智能体和环境的交互循环。

强化学习的主要角色是 **智能体** 和 **环境**，环境是智能体存在和交互的世界。
智能体在每一步的交互中，都会获得对于所处环境状态的观测（有可能只是一部分），然后决定下一步要执行的动作。
环境会因为智能体对它的动作而改变，也可能自己改变。

智能体也会从环境中感知到 **奖励** 信号，一个表明当前状态好坏的数字。
智能体的目标是最大化累计奖励，也就是 **回报**。强化学习就是智能体通过学习行为来实现目标的方法。

为了更具体地讨论强化的作用，我们需要引入其他术语。我们会讨论：

* 状态和观测，
* 动作空间，
* 策略，
* 轨迹，
* 不同的回报公式，
* 强化学习优化问题，
* 值函数。


状态和观测
-----------------------

一个 **状态** :math:`s` 是一个关于这个世界状态的完整描述。这个世界除了状态以外没有别的信息。
观测 :math:`o` 是对于一个状态的部分描述，可能会漏掉一些信息。

在深度强化学习中，我们一般用 `实数向量、矩阵或者更高阶的张量`_ 表示状态和观测。
比如说，视觉上的观测可以用用其像素值的RGB矩阵表示；机器人的状态可以通过关节角度和速度来表示。

如果智能体观测到环境的全部状态，我们通常说环境是被 **全面观测** 的。如果智能体只能观测到一部分，我们称之为被 **部分观测**。

.. admonition:: 你应该知道

    强化学习有时候用符号 :math:`s` 代表状态，有些地方也会写作观测符号 :math:`o`。
    尤其是，当智能体在决定采取什么动作的时候，符号上的表示按理动作是基于状态的，
    但实际上，动作是基于观测的，因为智能体并不能知道状态（只能通过观测了解状态）。

    在我们的教程中，我们会按照标准的方式使用这些符号，不过你一般能从上下文中看出来具体表示什么。
    如果你觉得有些内容不够清楚，请提出issue！我们的目的是教会大家，不是让大家混淆。

.. _`实数向量、矩阵或者更高阶的张量`: https://en.wikipedia.org/wiki/Real_coordinate_space


动作空间
-------------

不同的环境允许不同的动作。所有给定环境中有效动作的集合称之为 **动作空间**。
有些环境，比如说 Atari 游戏和围棋，属于 **离散动作空间**，这种情况下智能体只能采取有限的动作。
其他的一些环境，比如智能体在物理世界中控制机器人，属于 **连续动作空间**。在连续动作空间中，动作是实数向量。

这种区别对于深度强化学习来说，影响深远。有些种类的算法只能一种情况下直接使用，而在另一种情况下则必须进行大量修改。


策略
--------

**策略** 是智能体用于决定下一步执行什么动作的规则。可以是确定性的，一般表示为 :math:`\mu`：

.. math::

    a_t = \mu(s_t),

也可以是随机的，一般表示为 :math:`\pi`:

.. math::

    a_t \sim \pi(\cdot | s_t).

因为策略本质上就是智能体的大脑，所以很多时候“策略”和“智能体”这两个名词经常互换，例如我们会说：“策略的目的是最大化奖励”。

在深度强化学习中，我们处理的是 **参数化的策略**：策略的输出依赖于一系列计算函数，
而这些函数又依赖于参数（例如神经网络的权重和偏差），所以我们可以通过一些优化算法改变智能体的的行为。

我们经常把这些策略的参数写作 :math:`\theta` 或者 :math:`\phi` ，然后把它写在策略的下标上来强调两者的联系。

.. math::

    a_t &= \mu_{\theta}(s_t) \\
    a_t &\sim \pi_{\theta}(\cdot | s_t).

确定性策略
^^^^^^^^^^^^^^^^^^^^^^

**例子：确定性策略**。这是一个基于 TensorFlow 在连续动作空间上构建确定性策略的简单例子：

.. code-block:: python

    obs = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    net = mlp(obs, hidden_dims=(64,64), activation=tf.tanh)
    actions = tf.layers.dense(net, units=act_dim, activation=None)

其中，``mlp`` 是把多个给定大小和激活函数的 ``dense`` 相互堆积在一起的函数。

.. _stochastic-policies:

随机策略
^^^^^^^^^^^^^^^^^^^

深度强化学习中最常见的两种随机策略是 **类别策略 (Categorical Policies）** 和
**对角高斯策略 (Diagonal Gaussian Policies）**。

`类别`_ 策略适用于离散动作空间，而 `高斯`_ 策略一般用在连续动作空间。

对于使用和训练随机策略，两个关键计算至关重要：

* 从策略中采样动作，
* 计算特定行为的对数似然 :math:`\log \pi_{\theta}(a|s)`。

接下来，我们将描述针对类别策略和对角高斯策略如何执行这些操作。

.. admonition:: 类别策略

    类别策略就像是一个离散空间的分类器。对于分类器和确定策略来说，建立神经网络的方式一模一样：
    输入是观测，接着是若干层（可能是卷积或全连接层，具体取决于输入的类型），
    最后是一个线性层给出每个动作的 logit 值，后面跟一个 `softmax`_ 层把 logit 值转换为概率。

    **采样**。给定每个动作的概率，TensorFlow之类的框架有内置采样工具。
    具体可查阅 `tf.distributions.Categorical`_ 或者 `tf.multinomial`_ 文档.

    **对数似然**：表示最后一层的概率 :math:`P_{\theta}(s)`。
    它是一个有很多值的向量，我们可以把动作当做向量的索引。
    所以向量的对数似然值 :math:`a` 可以通过索引向量得到：

    .. math::

        \log \pi_{\theta}(a|s) = \log \left[P_{\theta}(s)\right]_a.


.. admonition:: 对角高斯策略

    多元高斯分布（或者多元正态分布），可以用一个均值向量 :math:`\mu` 和协方差 :math:`\Sigma` 来描述。
    对角高斯分布就是协方差矩阵只在对角线上有值的特殊情况，所以我们可以用一个向量来表示它。

    对角高斯策略总会有一个神经网络，表示观测到平均动作 :math:`\mu_{\theta}(s)` 通常有两种不同的方式表示协方差矩阵。

    **第一种方法**：有一个对数标准差的单独的向量 :math:`\log \sigma`，它不是关于状态的函数：:math:`\log \sigma` 是单独的参数。
    （你应该知道，我们对 VPG, TRPO 和 PPO 都是用这种方式实现的。）

    **第二种方法**：有一个神经网络，从状态映射到对数标准差 :math:`\log \sigma_{\theta}(s)`。
    它可以选择与均值网络共享某些层。

    要注意这两种情况下我们都没有直接计算标准差而是计算对数标准差。
    这是因为对数标准差能够接受 :math:`(-\infty, \infty)` 之间的任何值，而标准差必须要求参数非负。
    要知道，限制条件越少，训练就越简单。而标准差可以通过取幂快速从对数标准差中计算得到，所以这种表示方法也不会丢失信息。

    **采样**。给定平均动作 :math:`\mu_{\theta}(s)` 和标准差 :math:`\sigma_{\theta}(s)`，
    以及一个服从球形高斯分布（:math:`z \sim \mathcal{N}(0, I)`）的噪声向量 :math:`z`，动作样本可以这样计算：

    .. math::

        a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z,

    其中 :math:`\odot` 表示两个向量按元素相乘。标准框架都有内置噪声向量实现，例如 `tf.random_normal`_。
    你也可以直接用 `tf.distributions.Normal`_ 以均值和标准差的方式来采样。

    **对数似然**。一个基于均值为 :math:`\mu = \mu_{\theta}(s)`，
    标准差为 :math:`\sigma = \sigma_{\theta}(s)` 的对角高斯的 :math:`k` 维动作 action :math:`a` 的对数似然为：

    .. math::

        \log \pi_{\theta}(a|s) = -\frac{1}{2}\left(\sum_{i=1}^k \left(\frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i \right) + k \log 2\pi \right).


.. _`类别`: https://en.wikipedia.org/wiki/Categorical_distribution
.. _`高斯`: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
.. _`softmax`: https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
.. _`tf.distributions.Categorical`: https://www.tensorflow.org/api_docs/python/tf/distributions/Categorical
.. _`tf.multinomial`: https://www.tensorflow.org/api_docs/python/tf/multinomial
.. _`tf.random_normal`: https://www.tensorflow.org/api_docs/python/tf/random_normal
.. _`tf.distributions.Normal`: https://www.tensorflow.org/api_docs/python/tf/distributions/Normal


轨迹（Trajectories）
----------------------

轨迹 :math:`\tau` 指的是状态和动作的序列，

.. math::

    \tau = (s_0, a_0, s_1, a_1, ...).

第一个状态 :math:`s_0` 是从 **开始状态分布** 中随机采样的，有时候表示为 :math:`\rho_0`：

.. math::

    s_0 \sim \rho_0(\cdot).

转态转换（从某一时间 :math:`t` 的状态 :math:`s_t` 到
另一时间 :math:`t+1` 的状态 :math:`s_{t+1}` 会发生什么），
是由环境的自然法则确定的，并且只依赖于最近的动作 :math:`a_t`。它们可以是确定性的：

.. math::

    s_{t+1} = f(s_t, a_t)

也可以是随机的：

.. math::

    s_{t+1} \sim P(\cdot|s_t, a_t).

智能体的动作由策略确定。

.. admonition:: 你应该知道

    轨迹常常也被称作 **回合(episodes)** 或者 **rollouts**。

.. _reward-and-return:

奖励和回报
-----------------

强化学习中，奖励函数 :math:`R` 非常重要。它由当前状态、已经执行的动作和下一步的状态共同决定。

.. math::

    r_t = R(s_t, a_t, s_{t+1})

有时候这个公式会被改成只依赖当前的状态 :math:`r_t = R(s_t)`，或者状态动作对 :math:`r_t = R(s_t,a_t)`。

智能体的目标是最大化轨迹的累积奖励，这实际上意味着很多事情。我们会把所有的情况表示为 :math:`R(\tau)`，
至于具体表示什么，要么可以很清楚的从上下文看出来，要么并不重要。（因为相同的方程式适用于所有情况。）

一种回报是 **有限视野无折扣回报** （finite-horizon undiscounted return），指的是在一个固定窗口步数内获得的奖励之和：

.. math::

    R(\tau) = \sum_{t=0}^T r_t.

另一种回报是 **无限视野折扣回报** （infinite-horizon discounted return），指的是智能体 *曾经* 获得的全部奖励之和，
但是奖励会因为获得的时间不同而衰减。这个公式包含折扣因子 :math:`\gamma \in (0,1)`：

.. math::

    R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t.

这里为什么要加上一个折扣因子呢？为什么不直接把 *所有* 奖励加在一起？这么做，但是折扣因子在直观上和数学上都很方便。

直观上讲，现在的奖励比未来的奖励要好；数学角度上，无限多个奖励的和 `可能不能收敛`_ 到有限值，并且很难用方程来处理。
有了折扣因子和适当的约束条件，无穷和收敛。

.. admonition:: 你应该知道

    这两个公式在强化学习公式中看起来差距很大，但深度强化学习上经常会混用。
    比如说，我们经常使用算法以优化无折扣回报收益，但是用折扣因子估计 **值函数**。

.. _`可能不能收敛`: https://en.wikipedia.org/wiki/Convergent_series


强化学习问题
--------------

无论选择哪种方式衡量收益（有限视野无折扣回报或者无限视野折扣回报），无论选择哪种策略，
强化学习的目标都是选择一种策略从而最大化 **期望回报**。

讨论期望回报之前，我们先讨论下轨迹的概率分布。

我们假设环境转换和策略都是随机的。这种情况下，:math:`T` 步的轨迹是：

.. math::

    P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t).


期望回报（无论哪种方式衡量）:math:`J(\pi)` 是：

.. math::

    J(\pi) = \int_{\tau} P(\tau|\pi) R(\tau) = \underE{\tau\sim \pi}{R(\tau)}.


强化学习中的核心优化问题可以表示为：

.. math::

    \pi^* = \arg \max_{\pi} J(\pi),

:math:`\pi^*` 是 **最优策略**。

.. _value-functions:

值函数
---------------

知道一个状态的 **值** 或者状态动作对很有用。这里的值指的是，如果你从某一个状态或者状态动作对开始，
一直按照某个策略运行下去最终获得的期望回报。几乎是所有的强化学习算法，都在使用一种或另一种形式的 **值函数**。

这里介绍四种主要函数：

1. **同轨策略值函数**：:math:`V^{\pi}(s)`，从某一个状态 :math:`s` 开始，
之后每一步动作都按照策略 :math:`\pi` 执行的期望回报：

    .. math::

        V^{\pi}(s) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}

2. **同轨策略动作值函数**：:math:`Q^{\pi}(s,a)`，从某一个状态 :math:`s` 开始，
先随便执行一个动作 :math:`a` （有可能不是按照策略走的），之后每一步都按照策略 :math:`\pi` 执行的期望回报：

    .. math::

        Q^{\pi}(s,a) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}

3. **最优值函数**：:math:`V^*(s)`，从某一个状态 :math:`s` 开始，之后每一步都按照 *最优* 策略执行的期望回报：

    .. math::

        V^*(s) = \max_{\pi} \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}

4. **最优动作值函数**：:math:`Q^*(s,a)`，从某一个状态 :math:`s` 开始，
先随便执行一个动作 :math:`a`，之后每一步都按照 *最优* 策略执行的期望回报：

    .. math::

        Q^*(s,a) = \max_{\pi} \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}


.. admonition:: 你应该知道

    当我们讨论值函数的时候，如果我们没有提到时间依赖问题，我们仅指预期的 **无限视野折扣回报**。
    有限视野无折扣回报的值函数需要传入时间作为参数，你知道为什么吗？ 提示：时间到了会发生什么？

.. admonition:: 你应该知道

    值函数和动作值函数之间经常会出现两个关键联系：

    .. math::

        V^{\pi}(s) = \underE{a\sim \pi}{Q^{\pi}(s,a)},

    以及

    .. math::

        V^*(s) = \max_a Q^* (s,a).

    这些关系直接来自刚刚给出的定义，你能尝试给出证明吗？


最优 Q 函数和最优动作
---------------------------------------------

最优动作值函数 :math:`Q^*(s,a)` 和被最优策略选中的动作之间有重要的联系。
从定义上讲，:math:`Q^*(s,a)` 指的是从一个状态 :math:`s` 开始，执行（任意）一个行动 :math:`a`，
然后一直按照最优策略执行下去所获得的期望回报。

状态 :math:`s` 的最优策略会选择从状态 :math:`s` 开始能够最大化期望回报的行动。
所以如果我们有了 :math:`Q^*`，就可以通过下面的公式直接获得最优动作 :math:`a^*(s)`：

.. math::

    a^*(s) = \arg \max_a Q^* (s,a).

注意：可能会有多个动作能够最大化 :math:`Q^*(s,a)`，这种情况下，它们都是最优动作，最优策略可能会从中随机选择一个。
但是总会存在一个最优策略每一步选择动作的时候都是确定的。


贝尔曼方程
-----------------

所有四个值函数都遵循称为 **贝尔曼方程** 的特殊自洽方程。贝尔曼方程背后的基本思想是：

    起始点的值等于当前点预期值和下一个点的值之和。

同轨策略值函数的贝尔曼方程是

.. math::
    :nowrap:

    \begin{align*}
    V^{\pi}(s) &= \underE{a \sim \pi \\ s'\sim P}{r(s,a) + \gamma V^{\pi}(s')}, \\
    Q^{\pi}(s,a) &= \underE{s'\sim P}{r(s,a) + \gamma \underE{a'\sim \pi}{Q^{\pi}(s',a')}},
    \end{align*}

其中 :math:`s' \sim P` 是 :math:`s' \sim P(\cdot |s,a)` 的简写，
表明下一个状态 :math:`s'` 是按照转移规则从环境中采样得到的；
:math:`a \sim \pi` 是 :math:`a \sim \pi(\cdot|s)` 的简写；
:math:`a' \sim \pi` 是 :math:`a' \sim \pi(\cdot|s')` 的简写。

最优值函数的贝尔曼方程是

.. math::
    :nowrap:

    \begin{align*}
    V^*(s) &= \max_a \underE{s'\sim P}{r(s,a) + \gamma V^*(s')}, \\
    Q^*(s,a) &= \underE{s'\sim P}{r(s,a) + \gamma \max_{a'} Q^*(s',a')}.
    \end{align*}

同轨策略值函数和最优值函数的贝尔曼方程的最大区别是在动作上是否 :math:`\max`。
这表明智能体在选择下动作时，为了采取最优动作，他必须选择能获得最大值的动作。

.. admonition:: 你应该知道

    贝尔曼算子（Bellman backup）在强化学习中经常出现。
    一个状态或一个状态动作对的贝尔曼算子在贝尔曼方程的右边：奖励加下一个价值。

.. _advantage-functions:

优势函数
-------------------

强化学习中，有些时候我们不需要描述一个行动的绝对好坏，而只需要知道它相对于平均水平的有多好。
也就是说，我们只想知道一个行动的相对 **优势** 。这就是 **优势函数** 的概念。

一个服从策略 :math:`\pi` 的优势函数 :math:`A^{\pi}(s,a)`，
描述的是它在状态 :math:`s` 下采取动作 :math:`a` 比
根据 :math:`\pi(\cdot|s)` 随机选择一个动作好多少（假设之后一直服从策略 :math:`\pi`）。
数学上，优势函数的定义为：

.. math::

    A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).

.. admonition:: 你应该知道

    我们之后会继续谈论优势函数，它对于策略梯度方法非常重要。


（可选）数学形式
====================

我们已经非正式地讨论了智能体的环境，但是如果你深入研究，可能会发现这样的标准数学形式：
**马尔科夫决策过程** (Markov Decision Processes，MDPs)。
马尔科夫决策过程是一个5元组 :math:`\langle S, A, R, P, \rho_0 \rangle`，其中

* :math:`S` 是所有有效状态的集合，
* :math:`A` 是所有有效动作的集合，
* :math:`R : S \times A \times S \to \mathbb{R}` 是奖励函数，
  其中 :math:`r_t = R(s_t, a_t, s_{t+1})`，
* :math:`P : S \times A \to \mathcal{P}(S)` 是状态转移概率函数，
  其中 :math:`P(s'|s,a)` 是在状态 :math:`s` 下 采取动作 :math:`a` 转移到状态 :math:`s'` 的概率，
* :math:`\rho_0` 是开始状态的分布。

马尔科夫决策过程指的是服从 `马尔科夫性`_ 的系统： 状态转移只依赖与最近的状态和行动，而不依赖之前的历史数据。


.. _`马尔科夫性`: https://en.wikipedia.org/wiki/Markov_property
