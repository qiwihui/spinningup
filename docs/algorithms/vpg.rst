=======================
Vanilla Policy Gradient
=======================

.. contents:: 目录

.. _background:

背景
==========

（前一节：`强化学习介绍：第三部分`_）

.. _`强化学习介绍：第三部分`: ../spinningup/rl_intro3.html

策略梯度背后的关键思想是提高导致更高回报的动作的概率，并降低导致更低回报的动作的概率，直到你获得最佳策略。

速览
-----------

* VPG 是一种在轨策略算法。
* VPG可用于具有离散或连续动作空间的环境。
* VPG的Spinning Up实现支持与MPI并行化。

关键方程
-------------

令 :math:`\pi_{\theta}` 表示参数为 :math:`\theta` 的策略，
而 :math:`J(\pi_{\theta})` 表示有限视野无折扣的策略回报期望。
:math:`J(\pi_{\theta})` 的梯度为

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
        },

其中 :math:`\tau` 是轨迹，:math:`A^{\pi_{\theta}}` 是当前策略的优势函数。

策略梯度算法的工作原理是通过随机梯度提升策略性能来更新策略参数：

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

策略梯度实现通常基于无限视野折扣回报来计算优势函数估计，尽管其他情况下使用有限视野无折扣策略梯度公式。

探索与利用
----------------------------

VPG以在轨策略方式训练随机策略。这意味着它会根据其随机策略的最新版本通过采样操作来进行探索。
动作选择的随机性取决于初始条件和训练程序。在训练过程中，由于更新规则鼓励该策略利用已经发现的奖励，因此该策略通常变得越来越少随机性。
这可能会导致策略陷入局部最优状态。

伪代码
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Vanilla Policy Gradient Algorithm}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta_0$, initial value function parameters $\phi_0$
        \FOR{$k = 0,1,2,...$} 
        \STATE Collect set of trajectories ${\mathcal D}_k = \{\tau_i\}$ by running policy $\pi_k = \pi(\theta_k)$ in the environment.
        \STATE Compute rewards-to-go $\hat{R}_t$.
        \STATE Compute advantage estimates, $\hat{A}_t$ (using any method of advantage estimation) based on the current value function $V_{\phi_k}$.
        \STATE Estimate policy gradient as
            \begin{equation*}
            \hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
            \end{equation*}
        \STATE Compute policy update, either using standard gradient ascent,
            \begin{equation*}
            \theta_{k+1} = \theta_k + \alpha_k \hat{g}_k,
            \end{equation*}
            or via another gradient ascent algorithm like Adam.
        \STATE Fit value function by regression on mean-squared error:
            \begin{equation*}
            \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,
            \end{equation*}
            typically via some gradient descent algorithm.
        \ENDFOR
    \end{algorithmic}
    \end{algorithm}


文档
=============

.. autofunction:: spinup.vpg

保存的模型的内容
--------------------

记录的计算图包括：

========  ====================================================================
键        值
========  ====================================================================
``x``     Tensorflow placeholder for state input.
``pi``    Samples an action from the agent, conditioned on states in ``x``.
``v``     Gives value estimate for states in ``x``.
========  ====================================================================

可以通过以下方式访问此保存的模型

* 使用 `test_policy.py`_ 工具运行经过训练的策略，
* 或使用 `restore_tf_graph`_ 将整个保存的图形加载到程序中。

.. _`test_policy.py`: ../user/saving_and_loading.html#loading-and-running-trained-policies
.. _`restore_tf_graph`: ../utils/logger.html#spinup.utils.logx.restore_tf_graph


参考
==========

相关论文
---------------

- `Policy Gradient Methods for Reinforcement Learning with Function Approximation`_, Sutton et al. 2000
- `Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs`_, Schulman 2016(a)
- `Benchmarking Deep Reinforcement Learning for Continuous Control`_, Duan et al. 2016
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016(b)

.. _`Policy Gradient Methods for Reinforcement Learning with Function Approximation`: https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
.. _`Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs`: http://joschu.net/docs/thesis.pdf
.. _`Benchmarking Deep Reinforcement Learning for Continuous Control`: https://arxiv.org/abs/1604.06778
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438

为什么是这些论文？
-------------------

包含Sutton 2000是因为它是强化学习理论的永恒经典，并且包含了导致现代策略梯度的早期工作的参考。
之所以包括Schulman 2016（a），是因为第2章对策略梯度算法（包括伪代码）的理论进行了清晰的介绍。
Duan 2016是一份清晰的，最新的基准论文，显示了深度强化学习设置
（例如，以神经网络策略和Adam为优化器）中的vanilla policy gradient与其他深度强化算法的比较。
之所以包含Schulman 2016（b），是因为我们在VPG的实现中利用了
通用优势估计（Generalized Advantage Estimation）来计算策略梯度。

其他公开实现
----------------------------

- rllab_
- `rllib (Ray)`_

.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/pg
