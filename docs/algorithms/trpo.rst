================================
Trust Region Policy Optimization
================================

.. contents:: 目录

.. _background:

背景
==========

（前一节： `VPG背景`_）

.. _`VPG背景`: ../algorithms/vpg.html#background

TRPO通过采取最大的可以改进策略的步来更新策略，同时满足关于允许新旧策略接近的特殊约束。
约束用 `KL散度`_ 表示，KL散度是对概率分布之间的距离（但不完全相同）的一种度量。

这与常规策略梯度不同，后者使新策略和旧策略在参数空间中保持紧密联系。
但是，即使参数空间上看似很小的差异也可能在性能上产生很大的差异──因此，一个糟糕的步骤可能会使策略性能崩溃。
这使得使用大步长的vanilla policy gradients变得危险，从而损害了其采样效率。
TRPO很好地避免了这种崩溃，并且倾向于快速单调地提高性能。

.. _`KL散度`: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

速览
-----------

* TRPO是在轨算法。
* TRPOVPG可用于具有离散或连续动作空间的环境。
* TRPO的Spinning Up实现支持与MPI并行化。

关键方程
-------------

令 :math:`\pi_{\theta}` 表示参数为 :math:`\theta` 的策略，理论上的TRPO更新为：

.. math::

    \theta_{k+1} = \arg \max_{\theta} \; & {\mathcal L}(\theta_k, \theta) \\
    \text{s.t.} \; & \bar{D}_{KL}(\theta || \theta_k) \leq \delta

其中 :math:`{\mathcal L}(\theta_k, \theta)` 是 *替代优势*，
它使用旧策略中的数据来衡量策略 :math:`\pi_{\theta}` 与旧策略 :math:`\pi_{\theta_k}` 的相对性能：

.. math::

    {\mathcal L}(\theta_k, \theta) = \underE{s,a \sim \pi_{\theta_k}}{
        \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)
        },

:math:`\bar{D}_{KL}(\theta || \theta_k)` 是旧策略访问的各状态之间的策略之间的平均散度差异：

.. math::

    \bar{D}_{KL}(\theta || \theta_k) = \underE{s \sim \pi_{\theta_k}}{
        D_{KL}\left(\pi_{\theta}(\cdot|s) || \pi_{\theta_k} (\cdot|s) \right)
    }.

.. admonition:: 你应该知道

    当 :math:`\theta = \theta_k` 时，目标和约束都为零。
    此外，当 :math:`\theta = \theta_k` 时，约束相对于 :math:`\theta` 的梯度为零。
    要证明这些事实，需要对相关数学有一些微妙的掌握──每当您准备就绪时，这都是值得做的练习！

理论上的TRPO更新不是最容易使用的，因此TRPO做出了一些近似以快速获得答案。
我们使用泰勒展开将目标和约束扩展到 :math:`\theta_k` 周围的首阶指数（leading order）：

.. math::

    {\mathcal L}(\theta_k, \theta) &\approx g^T (\theta - \theta_k) \\
    \bar{D}_{KL}(\theta || \theta_k) & \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)

结果产生一个近似的优化问题，

.. math::

    \theta_{k+1} = \arg \max_{\theta} \; & g^T (\theta - \theta_k) \\
    \text{s.t.} \; & \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta.

.. admonition:: 你应该知道

    巧合的是，以 :math:`\theta = \theta_k` 评估的替代优势函数相对于 :math:`\theta` 的梯度 :math:`g`
    恰好等于策略梯度 :math:`\nabla_{\theta} J(\pi_{\theta})`！
    如果您愿意精通数学，请尝试证明这一点。

这个近似问题可以通过拉格朗日对偶 [1]_ 的方法来解析地解决，得出的结果是：

.. math::

    \theta_{k+1} = \theta_k + \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g.

如果我们到此为止，并仅使用此最终结果，该算法将准确地计算 `自然策略梯度`_ （Natural Policy Gradient）。
一个问题是，由于泰勒展开式引入的近似误差，这可能无法满足KL约束，或实际上提高了替代优势。
TRPO对此更新规则进行了修改：回溯行搜索，

.. math::

    \theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g,

其中 :math:`\alpha \in (0,1)` 是回溯系数，
:math:`j` 是 :math:`\pi_{\theta_{k+1}}` 满足KL约束并产生正的替代优势的最小非负整数。

Lastly: computing and storing the matrix inverse, :math:`H^{-1}`, is painfully expensive when dealing with neural network policies with thousands or millions of parameters.
TRPO sidesteps the issue by using the `conjugate gradient`_ algorithm to solve :math:`Hx = g` for :math:`x = H^{-1} g`, 
requiring only a function which can compute the matrix-vector product :math:`Hx` instead of computing and storing the whole matrix :math:`H` directly. 
This is not too hard to do: we set up a symbolic operation to calculate

最后：处理带有成千上万个参数的神经网络策略时，矩阵逆 :math:`H^{-1}` 的计算和存储非常昂贵。
TRPO通过使用 `共轭梯度`_ 算法对 :math:`x = H^{-1} g` 求解 :math:`Hx = g` 来回避问题，
仅需要一个可以计算矩阵矢量乘积 :math:`Hx` 的函数，而不是直接计算和存储整个矩阵 :math:`H`。
这并不难：我们设置了一个符号运算来计算

.. math::

    Hx = \nabla_{\theta} \left( \left(\nabla_{\theta} \bar{D}_{KL}(\theta || \theta_k)\right)^T x \right),

这样就可以在不计算整个矩阵的情况下提供正确的输出。

.. [1] 参见Boyd和Vandenberghe的 `凸优化`_，特别是第2至第5章。

.. _`凸优化`: http://stanford.edu/~boyd/cvxbook/
.. _`自然策略梯度`: https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf
.. _`共轭梯度`: https://en.wikipedia.org/wiki/Conjugate_gradient_method

探索与利用
----------------------------

TRPO trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.

TRPO以一种在轨方式训练随机策略。这意味着它会根据其随机策略的最新版本通过采样操作来进行探索。
动作选择的随机性取决于初始条件和训练程序。
在训练过程中，由于更新规则鼓励该策略利用已经发现的奖励，因此该策略通常变得越来越少随机性。
这可能会导致策略陷入局部最优状态。

伪代码
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Trust Region Policy Optimization}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta_0$, initial value function parameters $\phi_0$
        \STATE Hyperparameters: KL-divergence limit $\delta$, backtracking coefficient $\alpha$, maximum number of backtracking steps $K$
        \FOR{$k = 0,1,2,...$} 
        \STATE Collect set of trajectories ${\mathcal D}_k = \{\tau_i\}$ by running policy $\pi_k = \pi(\theta_k)$ in the environment.
        \STATE Compute rewards-to-go $\hat{R}_t$.
        \STATE Compute advantage estimates, $\hat{A}_t$ (using any method of advantage estimation) based on the current value function $V_{\phi_k}$.
        \STATE Estimate policy gradient as
            \begin{equation*}
            \hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
            \end{equation*}
        \STATE Use the conjugate gradient algorithm to compute
            \begin{equation*}
            \hat{x}_k \approx \hat{H}_k^{-1} \hat{g}_k,
            \end{equation*}
            where $\hat{H}_k$ is the Hessian of the sample average KL-divergence.
        \STATE Update the policy by backtracking line search with
            \begin{equation*}
            \theta_{k+1} = \theta_k + \alpha^j \sqrt{ \frac{2\delta}{\hat{x}_k^T \hat{H}_k \hat{x}_k}} \hat{x}_k,
            \end{equation*}
            where $j \in \{0, 1, 2, ... K\}$ is the smallest value which improves the sample loss and satisfies the sample KL-divergence constraint.
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

.. autofunction:: spinup.trpo

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

- `Trust Region Policy Optimization`_, Schulman et al. 2015
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016
- `Approximately Optimal Approximate Reinforcement Learning`_, Kakade and Langford 2002

.. _`Trust Region Policy Optimization`: https://arxiv.org/abs/1502.05477
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438
.. _`Approximately Optimal Approximate Reinforcement Learning`: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf

为什么是这些论文？
--------------------

包含Schulman 2015是因为它是描述TRPO的原始论文。
之所以包含Schulman 2016，是因为我们对TRPO的实现利用了通用优势估计来计算策略梯度。
之所以将Kakade和Langford 2002包括在内是因为它包含的理论结果激励并深深地与TRPO的理论基础联系在一起。

其他公开实现
----------------------------

- Baselines_
- ModularRL_
- rllab_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/trpo_mpi
.. _ModularRL: https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/trpo.py
