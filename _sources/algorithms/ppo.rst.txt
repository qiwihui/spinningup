============================
Proximal Policy Optimization
============================

.. contents:: Table of Contents


背景
==========

(前一节： `TRPO背景`_)

.. _`TRPO背景`: ../algorithms/trpo.html#background

PPO受到与TRPO相同的问题的激励：我们如何才能使用当前拥有的数据在策略上采取最大可能的改进步骤，
而又不会走得太远而导致意外使性能下降？在TRPO尝试使用复杂的二阶方法解决此问题的地方，
PPO是一阶方法的族，它们使用其他一些技巧来使新策略接近于旧策略。
PPO方法明显更易于实现，并且从经验上看，其性能至少与TRPO相同。

PPO有两种主要变体：PPO-Penalty和PPO-Clip。

**PPO-Penalty** 近似解决了TRPO之类的受KL约束的更新，但是惩罚了目标函数中的KL背离而不是使其成为硬约束，
并且在训练过程中自动调整了惩罚系数，以便适当地缩放。

**PPO-Clip** 在目标中没有KL散度项，也没有任何约束。取而代之的是依靠对目标函数的专门削减来消除新策略远离旧策略的动机。

在这里，我们仅关注PPO-Clip（OpenAI使用的主要变体）。

速览
-----------

* PPO是在轨算法。
* PPO可用于具有离散或连续动作空间的环境。
* PPO的Spinning Up实现支持与MPI并行化。

关键方程
-------------

PPO-clip 通过以下更新策略

.. math::

    \theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[
        L(s,a,\theta_k, \theta)\right],

通常采取多个步骤（通常是小批量）SGD来最大化目标。这里 :math:`L` 是由

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)
    \right),

其中 :math:`\epsilon` 是一个（小）超参数，它粗略地说出了新策略与旧策略的距离。

这是一个非常复杂的表述，很难一眼就知道它在做什么，或者它如何帮助使新策略接近旧策略。
事实证明，此目标有一个相当简化的版本 [1]_，它易于处理（也是我们在代码中实现的版本）：

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    g(\epsilon, A^{\pi_{\theta_k}}(s,a))
    \right),

其中

.. math::

    g(\epsilon, A) = \left\{
        \begin{array}{ll}
        (1 + \epsilon) A & A \geq 0 \\
        (1 - \epsilon) A & A < 0.
        \end{array}
        \right.

为了弄清楚从中得到的直觉，让我们看一个状态对 :math:`(s,a)`，并分情况考虑。


**优势是正的**：假设该状态-动作对的优势是正的，在这种情况下，它对目标的贡献减少为

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 + \epsilon)
    \right)  A^{\pi_{\theta_k}}(s,a).

因为优势是正的，所以如果采取行动的可能性更大，也就是说，如果 :math:`\pi_{\theta}(a|s)` 增加，则目标也会增加。
但是此术语中的最小值限制了目标可以增加的 *程度*。
一旦 :math:`\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)`，最小值就会增加，
此项达到 :math:`(1+\epsilon) A^{\pi_{\theta_k}}(s,a)` 的上限 。
因此：*远离旧策略不会使新政策受益*。

**优势是负的**：假设该状态-动作对的优势是负的，在这种情况下，它对目标的贡献减少为

.. math::

    L(s,a,\theta_k,\theta) = \max\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 - \epsilon)
    \right)  A^{\pi_{\theta_k}}(s,a).

因为优势是负的，所以如果行动变得不太可能（即 :math:`\pi_{\theta}(a|s)` 减小），则目标将增加。
但是此术语中的最大值限制了可以增加的 *程度*。
一旦 :math:`\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)`，最大值就会增加，
此项达到 :math:`(1-\epsilon) A^{\pi_{\theta_k}}(s,a)` 的上限。
因此，再次：*新政策不会因远离旧政策而受益*。

到目前为止，我们已经看到剪裁通过消除策略急剧变化的诱因而成为一种调节器，
而超参数 :math:`\epsilon` 对应于新策略与旧策略的距离的远近，同时仍然有利于实现目标。

.. admonition:: 你应该知道

    尽管这种削减对确保合理的策略更新大有帮助，但仍然有可能最终产生与旧策略相距太远的新策略，
    并且不同的PPO实现使用很多技巧来避免这种情况。在此处的实现中，我们使用一种特别简单的方法：提前停止。
    如果新策略与旧策略的平均KL散度差距超出阈值，我们将停止采取梯度步骤。

    如果你对基本的数学知识和实施细节感到良好，则有必要查看其他实施以了解它们如何处理此问题！

.. [1] 请参阅 `此说明`_，以简化PPO-Clip目标的形式。

.. _`此说明`: https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view?usp=sharing

探索与利用
----------------------------

PPO以一种在轨策略方式训练随机策略。这意味着它会根据其随机策略的最新版本通过采样操作来进行探索。
动作选择的随机性取决于初始条件和训练程序。
在训练过程中，由于更新规则鼓励该策略利用已经发现的奖励，因此该策略通常变得越来越少随机性。
这可能会导致策略陷入局部最优状态。

伪代码
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{PPO-Clip}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta_0$, initial value function parameters $\phi_0$
        \FOR{$k = 0,1,2,...$} 
        \STATE Collect set of trajectories ${\mathcal D}_k = \{\tau_i\}$ by running policy $\pi_k = \pi(\theta_k)$ in the environment.
        \STATE Compute rewards-to-go $\hat{R}_t$.
        \STATE Compute advantage estimates, $\hat{A}_t$ (using any method of advantage estimation) based on the current value function $V_{\phi_k}$.
        \STATE Update the policy by maximizing the PPO-Clip objective:
            \begin{equation*}
            \theta_{k+1} = \arg \max_{\theta} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \min\left(
                \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}  A^{\pi_{\theta_k}}(s_t,a_t), \;\;
                g(\epsilon, A^{\pi_{\theta_k}}(s_t,a_t))
            \right),
            \end{equation*}
            typically via stochastic gradient ascent with Adam.
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

.. autofunction:: spinup.ppo

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

- `Proximal Policy Optimization Algorithms`_, Schulman et al. 2017
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016
- `Emergence of Locomotion Behaviours in Rich Environments`_, Heess et al. 2017

.. _`Proximal Policy Optimization Algorithms`: https://arxiv.org/abs/1707.06347
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438
.. _`Emergence of Locomotion Behaviours in Rich Environments`: https://arxiv.org/abs/1707.02286

为什么是这些论文？
--------------------

包含Schulman 2017是因为它是描述PPO的原始论文。
之所以包含Schulman 2016，是因为我们对PPO的实现利用了通用优势估计来计算策略梯度。
包含了Heess 2017，因为它提供了对复杂环境中PPO代理所学行为的大规模实证分析（尽管它使用PPO-Penalty而不是PPO-clip）。

其他公开实现
----------------------------

- Baselines_
- ModularRL_ （注意：这个实现了PPO-penalty而不是PPO-clip。）
- rllab_ （注意：这个实现了PPO-penalty而不是PPO-clip。）
- `rllib (Ray)`_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/ppo2
.. _ModularRL: https://github.com/joschu/modular_rl/blob/master/modular_rl/ppo.py
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/ppo.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ppo
