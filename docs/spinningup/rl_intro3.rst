====================================
第三部分：策略优化介绍
====================================

.. contents:: 目录
    :depth: 2


在这个部分，我们会讨论策略优化算法的数学基础，同时提供样例代码。我们会包括 **策略梯度** 理论的三个关键结果：

* `最简等式`_ 描述相对于策略参数的策略性能的梯度，
* 一条允许我们从该表达式中 `删除无用的术语`_ 的规则，
* 以及允许我们在该表达式中 `添加有用的术语`_ 的规则。

最后，我们会把结果放在一起，然后描述策略梯度基于优势函数的版本：我们在 `Vanilla Policy Gradient`_ 实现中使用的版本。

.. _`最简等式`: ../spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
.. _`删除无用的术语`: ../spinningup/rl_intro3.html#don-t-let-the-past-distract-you
.. _`添加有用的术语`: ../spinningup/rl_intro3.html#baselines-in-policy-gradients
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html

.. _deriving-the-simplest-policy-gradient:

推导最简单的策略梯度
=====================================

我们考虑一种随机的参数化策略 :math:`\pi_{\theta}`。
我们的目标是最大化期望回报 :math:`J(\pi_{\theta})=\underE{\tau \sim \pi_{\theta}}{R(\tau)}`。
出于方便推导，我们假定 :math:`R(\tau)` 是 `有限视野无折扣回报`_，对于无限视野折扣回报的推导几乎是相同。

.. _`有限视野无折扣回报`: ../spinningup/rl_intro.html#reward-and-return

我们将通过梯度下降来优化策略，例如

.. math::

    \theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.

策略性能的梯度 :math:`\nabla_{\theta} J(\pi_{\theta})` 称为 **策略梯度**，
而以这种方式优化策略的算法称为 **策略梯度算法**。
（例如Vanilla Policy Gradient和TRPO。PPO通常称为策略梯度算法，尽管这有点不准确。）

要实际使用此算法，我们需要一个可以通过数值计算的策略梯度表达式。这涉及两个步骤：
1）得出策略性能的解析梯度，证明其具有期望值的形式，
2）形成该期望值的样本估计，可以使用有限数量的智能体与环境相互作用的步骤数据计算得出。

在本小节中，我们将找到该表达式的最简形式。
在后面的小节中，我们将展示如何以最简的形式进行改进，以得到我们在标准策略梯度实现中实际使用的版本。

我们将首先列出一些事实，这些事实对于推导解析梯度非常有用。

**1. 轨迹的概率**。由 :math:`\pi_{\theta}` 给出的动作的轨迹 :math:`\tau = (s_0, a_0, ..., s_{T+1})` 的概率为：

.. math::

    P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).

**2. 对数导数技巧**。对数导数技巧基于微积分的一条简单规则：:math:`\log x` 相对于 :math:`x` 的导数为 :math:`1/x`。
重新排列并与链式规则结合后，我们得到：

.. math::

    \nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).

**3. 轨迹的对数概率**。 轨迹的对数概率为

.. math::

    \log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).

**4. 环境方程的梯度**。环境不依赖于 :math:`\theta`，
所以 :math:`\rho_0(s_0)`， :math:`P(s_{t+1}|s_t, a_t)` 和 :math:`R(\tau)` 的梯度为零。

**5. 轨迹对数概率的梯度**。轨迹对数概率的梯度为：

.. math::

    \nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
    &= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).

综上所述，我们得出以下结论：

.. admonition:: 基本策略梯度的推导

    .. math::
        :nowrap:

        \begin{align*}
        \nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \underE{\tau \sim \pi_{\theta}}{R(\tau)} & \\
        &= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
        &= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
        &= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
        &= \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log P(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
        \therefore \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob}
        \end{align*}

这是一个期望，这意味着我们可以使用样本均值对其进行估计。
如果我们收集一组轨迹 :math:`\mathcal{D} = \{\tau_i\}_{i=1,...,N}`，
其中每一个轨迹通过让智能体在环境中使用策略 :math:`\pi_{\theta}` 执行操作得到，则策略梯度可以使用以下式子进行估计：

.. math::

    \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),

其中 :math:`|\mathcal{D}|` 是 :math:`\mathcal{D}` 中轨迹的数量（在这里为 :math:`N`）。

最后一个表达式是我们想要的可计算表达式的最简单版本。
假设我们以允许我们计算 :math:`\nabla_{\theta} \log \pi_{\theta}(a|s)` 的方式表示我们的策略，
并且如果我们能够在环境中运行该策略以收集轨迹数据，则我们可以计算策略梯度并采取更新步骤。


实现最简单的策略梯度
=========================================

我们在 ``spinup/examples/pg_math/1_simple_pg.py`` 中给出了此简单版本的策略梯度算法的简短Tensorflow实现。
（也可以 `在github上 <https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py>`_ 查看。）
只有122行，因此我们强烈建议你深入阅读。虽然我们不会在这里介绍全部代码，但我们将重点介绍一些重要的部分。

**1. 建立策略网络**。

.. code-block:: python
    :linenos:
    :lineno-start: 25

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

该代码快构建了前馈神经网络分类策略。（新手请参见第一部分 `随机策略`_ 一节。）
``logits`` 张量可用于构造对数概率和动作概率，``actions`` 张量根据 ``logits`` 所隐含的概率对动作进行采样。

.. _`随机策略`: ../spinningup/rl_intro.html#stochastic-policies

**2. 构造损失函数**。

.. code-block:: python
    :linenos:
    :lineno-start: 32

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)

在此块中，我们为策略梯度算法构建“损失”函数。当插入正确的数据时，此损失的梯度等于策略梯度。
正确的数据表示根据当前策略执行操作时收集的一组（状态，动作，权重）元组，其中状态-动作对的权重是它所属episode的回报。
（你可以插入其他权重数据来使其正常工作，我们将在后面的小节中展示。）

.. admonition:: 你应该知道

    经管我们将其描述为损失函数，但从监督学习的角度来看，它并 **不是** 典型的损失函数。与标准损失函数有两个主要区别。

    **1. 数据分布取决于参数**。损失函数通常在固定的数据分布上定义，该分布与我们要优化的参数无关。
    这里不是这样，必须在最新策略上对数据进行采样。

    **2.它无法衡量效果**。损失函数通常会评估我们关注的性能指标。
    在这里，我们关心期望收益 :math:`J(\pi_{\theta})`，但即使在期望中，我们的“损失”函数也根本不近似。
    此“损失”函数仅对我们有用，因为当在当前参数下进行评估时，使用当前参数生成的数据，它的性能会呈现负梯度。

    但是，在梯度下降的第一步之后，它就不再与性能相关。这意味着，对于给定的一批数据，最小化此“损失”函数无法保证提高期望收益。
    你可以将这一损失设为 :math:`-\infty`，而策略性能可能下降。实际上，通常会这样。
    有时，资深强化学习研究人员可能将此结果描述为对大量数据“过度拟合”的策略。这是描述性的，但不应从字面上理解，因为它没有涉及泛化误差。

    之所以提出这一点，是因为机器学习练习者通常会在训练过程中将损失函数解释为有用的信号──“如果损失减少了，一切都会好起来的。”
    在政策梯度中，这种直觉是错误的，您应该只关心平均回报率。损失函数没有任何意义。

.. admonition:: 你应该知道

    此处用于生成 ``log_probs`` 张量的方法（创建操作掩码，并使用它来选择特定的对数概率） *仅* 适用于分类策略。通常它不起作用。

**3. 进行一个轮次的训练**。

.. code-block:: python
    :linenos:
    :lineno-start: 45

        # for training policy
        def train_one_epoch():
            # make some empty lists for logging.
            batch_obs = []          # for observations
            batch_acts = []         # for actions
            batch_weights = []      # for R(tau) weighting in policy gradient
            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            # reset episode-specific variables
            obs = env.reset()       # first obs comes from starting distribution
            done = False            # signal from environment that episode is over
            ep_rews = []            # list for rewards accrued throughout ep

            # render first episode of each epoch
            finished_rendering_this_epoch = False

            # collect experience by acting in the environment with current policy
            while True:

                # rendering
                if not(finished_rendering_this_epoch):
                    env.render()

                # save obs
                batch_obs.append(obs.copy())

                # act in the environment
                act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
                obs, rew, done, _ = env.step(act)

                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)

                if done:
                    # if episode is over, record info about episode
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

                    # reset episode-specific variables
                    obs, done, ep_rews = env.reset(), False, []

                    # won't render again this epoch
                    finished_rendering_this_epoch = True

                    # end experience loop if we have enough of it
                    if len(batch_obs) > batch_size:
                        break

            # take a single policy gradient update step
            batch_loss, _ = sess.run([loss, train_op],
                                     feed_dict={
                                        obs_ph: np.array(batch_obs),
                                        act_ph: np.array(batch_acts),
                                        weights_ph: np.array(batch_weights)
                                     })
            return batch_loss, batch_rets, batch_lens

``train_one_epoch()`` 函数运行一个策略梯度的“轮次”，我们定义为

1) 经验收集步骤（L62-97），其中智能体使用最新策略在环境中执行一定数量的episodes，然后是
2) 单个策略梯度更新步骤（L99-105）。

算法的主循环只是反复调用 ``train_one_epoch()``。


期望梯度对数概率引理
============================

在本小节中，我们将得出一个中间结果，该结果在整个策略梯度理论中得到了广泛使用。我们将其称为“期望梯度对数概率（EGLP）”引理。 [1]_

**EGLP 引理** 假设 :math:`P_{\theta}` 是随机变量 :math:`x` 上的参数化概率分布，则：

.. math::

    \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)} = 0.

.. admonition:: 证明

    我们知道所有概率分布均已归一化：

    .. math::

        \int_x P_{\theta}(x) = 1.

    取标归一形式的两侧的梯度：

    .. math::

        \nabla_{\theta} \int_x P_{\theta}(x) = \nabla_{\theta} 1 = 0.

    使用对数导数技巧可以得到：

    .. math::

        0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
        &= \int_x \nabla_{\theta} P_{\theta}(x) \\
        &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
        \therefore 0 &= \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)}.

.. [1] 本文的作者不知道是否在文献中的任何地方给该引理指定了标准名称。但是考虑到它出现的频率，似乎很值得给它起一个名字以便于参考。

.. _don-t-let-the-past-distract-you:

不要让过去使你分心
===============================

回顾我们对策略梯度的最新表达：

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}.

沿着这个梯度迈出一步，将每个动作的对数概率与 :math:`R(\tau)` 成正比，:math:`R(\tau)` 是 **曾经获得的所有奖励** 之和。 但这没有多大意义。

智能体实际上仅应根据其 *结果* 强化动作。采取动作之前获得的奖励与该动作的效果无关：只有 *获得的* 奖励。

事实证明，这种直觉体现在数学上，我们可以证明策略梯度也可以表示为

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}.

在这种形式中，仅根据采取动作后获得的奖励来强化动作。

我们将这种形式称为“reward-to-go策略梯度”，因为轨迹上某点之后的奖励总和，

.. math::

    \hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),

被称为从那点起的 **reward-to-go行的奖励**，而这种策略梯度表达式取决于状态动作对的reward-to-go。

.. admonition:: 你应该知道

    **但这如何更好？** 策略梯度的关键问题是需要多少个样本轨迹才能获得它们的低方差样本估计。
    我们从公式开始就包括了与过去的奖励成比例的强化动作的项，
    所有这些均值为零，但方差不为零：结果，它们只会给政策梯度的样本估计值增加噪音。
    通过删除它们，我们减少了所需的样本轨迹数量。

可以在 `此处`_ 找到该声明的（可选）证明，当然它基于EGLP引理。

.. _`此处`: ../spinningup/extra_pg_proof1.html


实现 Reward-to-Go 策略梯度
=========================================

我们在 ``spinup/examples/pg_math/2_rtg_pg.py`` 中给出了 reward-to-go 策略梯度算法的简短Tensorflow实现。
（也可以 `在 github 上 <https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py>`_ 查看。）

与 ``1_simple_pg.py`` 唯一不同的是，我们现在在损失函数中使用了不同的权重。
代码修改非常小：我们添加了一个新函数，并更改了另外两行。新函数是：

.. code-block:: python
    :linenos:
    :lineno-start: 12

    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

然后我们从以下方法调整旧的L86-87：

.. code-block:: python
    :linenos:
    :lineno-start: 86

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

为：

.. code-block:: python
    :linenos:
    :lineno-start: 93

                    # the weight for each logprob(a_t|s_t) is reward-to-go from t
                    batch_weights += list(reward_to_go(ep_rews))

.. _baselines-in-policy-gradients:

策略梯度基准
=============================

EGLP引理的直接结果是，对于仅依赖状态的任何函数 :math:`b`，

.. math::

    \underE{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)} = 0.

这使我们能够从我们的策略梯度表达式中加上或减去任何数量的这样的项，而无需更改它：

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}.

这种方式使用的任何函数 :math:`b` 称为 **基准**。

基线的最常见选择是 `同轨策略值函数`_ :math:`V^{\pi}(s_t)`。
回想一下，这是智能体从状态 :math:`s_t` 开始并在其余下的时间里按照策略 :math:`\pi` 执行动作所获得的平均回报。

根据经验，选择 :math:`b(s_t) = V^{\pi}(s_t)` 具有减少策略梯度样本估计中的方差的理想效果。
这样可以更快，更稳定地学习策略。从概念的角度来看，它也很有吸引力：它编码了一种直觉，即如果一个智能体获得了它预期的，它将“感觉”到中立。

.. admonition:: 你应该知道

    实际上，无法精确计算 :math:`V^{\pi}(s_t)` 因此必须将其近似。
    通常，这是通过神经网络 :math:`V_{\phi}(s_t)` 来完成的，该神经网络会与策略同时进行更新（以便价值网络始终近似于最新策略的值函数）。

    大多数策略优化算法（包括VPG，TRPO，PPO和A2C）的实现中使用的最简单的学习 :math:`V_{\phi}` 的方法是最小化均方误差：

    .. math:: \phi_k = \arg \min_{\phi} \underE{s_t, \hat{R}_t \sim \pi_k}{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2},

    | 其中 :math:`\pi_k` 是轮次 :math:`k` 的梯度。从先前的值参数 :math:`\phi_{k-1}` 开始，使用一个或多个梯度下降步骤完成此操作。


其他形式的策略梯度
==================================

到目前为止，我们看到的是策略梯度具有一般形式

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t},

其中 :math:`\Phi_t` 可以是

.. math:: \Phi_t &= R(\tau), 

或者

.. math:: \Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}), 

或者

.. math:: \Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t).

尽管有不同的差异，所有这些选择都导致相同的策略梯度期望值。事实证明，有两个权重 :math:`\Phi_t` 有效选择非常重要。

**1. 同轨动作值函数**。选择

.. math:: \Phi_t = Q^{\pi_{\theta}}(s_t, a_t)

也有效。有关此声明的（可选）证明，请参见 `此页面`_。

**2. 优势函数**。
回想一下 `动作的优势`_，定义为 :math:`A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)`，
描述相对于其他动作，平均而言（相对于当前策略）的好坏。这个选择

.. math:: \Phi_t = A^{\pi_{\theta}}(s_t, a_t)

也是有效的。证明是，它等同于使用 :math:`\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)` 然后使用值函数基线，
我们始终可以这么做。

.. admonition:: 你应该知道

    具有优势函数的策略梯度的公式极为普遍，并且有许多不同的方法来估算不同算法使用的优势函数。

.. admonition:: 你应该知道

    要对此主题进行更详细的处理，您应该阅读有关 `广义优势估计`_ （Generalized Advantage Estimation，GAE）的文章，
    该文章深入介绍了背景部分中 :math:`\Phi_t` 的不同选择。

    然后，该论文继续描述GAE，GAE是一种在策略优化算法中具有广泛用途的近似优势函数的方法。
    例如，Spinning Up的VPG，TRPO和PPO的实现都利用了它。因此，我们强烈建议你进行研究。


概括
=====

在本章中，我们描述了策略梯度方法的基本理论，并将一些早期结果与代码示例相关联。
有兴趣的学生应该从这里继续研究以后的结果（价值函数基准和策略梯度的优势公式）
如何转化为Spinning Up的 `Vanilla Policy Gradient`_ 的实现。

.. _`同轨策略值函数`: ../spinningup/rl_intro.html#value-functions
.. _`动作的优势`: ../spinningup/rl_intro.html#advantage-functions
.. _`此页面`: ../spinningup/extra_pg_proof2.html
.. _`广义优势估计`: https://arxiv.org/abs/1506.02438
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html
