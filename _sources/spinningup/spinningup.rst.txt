===================================
深度强化学习研究者资料
===================================
By Joshua Achiam, October 13th, 2018


.. contents:: 目录
    :depth: 2

如果你是一位深度强化学习的研究者，你现在可能已经对深度强化学习有了很多的了解。
你知道 `它很难不总是有效`_ 。即便是严格按照步骤来，`可重现性`_ 依然是 `一大挑战`_。
如果你准备从头开始，`学习的曲线非常陡峭`_ 。
虽然已经有 `很多`_ `很棒的`_ `学习`_ `资源`_，但是因为很多资料都很新，以至于还没有一条清晰明确的捷径。
这个项目的目的就是帮助你克服这些一开始的障碍，并且让你清楚的知道，如何成为一名深度强化学习研究员。
在这个项目里，我们会介绍一些有用的课程，作为基础知识，同时把一些可能适合研究的方向结合进来。


正确的背景
====================

**建立良好的数学背景**。从概率和统计学的角度，要对于随机变量、贝叶斯定理、链式法则、期望、标准差和重要性抽样等要有很好的理解。
从多重积分的角度，要了解梯度和泰勒展开（可选，但是会很有用）。

**对于深度学习要有基础的了解**。你不用知道每一个技巧和结构，但是了解基础的知识很有帮助。
要了解 `多层感知机`_、`vanilla RNN`_，`LSTM`_ （`同时看这篇博客`_）、`GRU`_、`卷积`_、
`层`_、`残差网络`_、`注意力`_ `机制`_，常见的正则手段（`weight decay`_，`dropout`_），
归一化方式(`batch norm`_，`layer norm`_，`weight norm`_)
和优化方式(`SGD, momentum SGD`_，`Adam`_，`以及其它`_)。
要了解什么是 `reparameterization trick`_ 。

**至少熟悉一种深度学习框架**。 `Tensorflow`_ 或者 `PyTorch`_ 非常适合练手。
你不用知道所有东西，但是你要能非常自信地实现一种监督学习算法。

**对于强化学习中的主要概念和术语很了解**。知道什么是状态、动作、轨迹、策略、奖励、值函数和动作值函数。
如果你对这些不了解，去读一读项目里面 `介绍`_ 部分的材料。
OpenAI Hackthon 的 `强化学习介绍`_ 也很值得看，或者是 `Lilian Weng 的综述`_。
如果你对于数学理论很感兴趣，可以学习 `单调提升理论`_ （高级策略梯度算法的的基础）或者 `经典强化学习算法`_
（尽管被深度强化学习所替代，但还是有很多能推动新的研究的洞见）。

.. _learn-by-doing:

在动手中学习
==============

**自己实现算法**。你应该尽可能地从头开始编写尽可能多的深度强化学习的核心算法，同时要保证自己的实现尽量简单、正确。这是了解这些算法如何工作、培养特定性能特征的直觉的最佳方法。

**简单是最重要的**。你要对自己的工作有合理的规划，从最简单的算法开始，然后慢慢引入复杂性。
如果你一开始就构建很多复杂的部分，有可能会耗费你接下来几周的时间来尝试调试。
对于刚刚接触强化学习的人来说，这是很常见的问题。如果你发现自己被困在其中，
不要气馁，尝试回到最开始然后换一种更简单的算法。

**哪些算法**？你可以大概按照 vanilla policy gradient(也被称为 `REINFORCE`_ )、`DQN`_，
`A2C`_ ( `A3C`_ 的同步版本)，`PPO`_ (具有 clipped objective 特性的变体), `DDPG`_ 的顺序来学习。
这些算法的最简版本可以用几百行代码编写（大约250-300行），有些更少，比如 `简洁版本的VPG`_ 只需要 80 行的代码。
在写并行版本代码之前，先尝试写单线程版本的。（至少实现一种并行的算法）

**注重理解**。编写有效的强化学习代码需要对于算法有明确的理解，同时注重细节。
因为 **错误的代码总是悄无声息**：看起来运行的很正常，但实际上智能体什么也没有学到。
这种情况通常是因为有些公式写错了，或者分布不对，又或者数据传输到了错误的地方。
有时候找到这些错误的唯一办法，就是批判性地阅读代码，明确知道它应该做什么，找到它偏离正确行为的地方。
这就需要你一方面了解学术文献，另一方面参考已有的实现，所以你要花很多时间在这些上面。

**看论文的时候要注意什么**：当基于论文实现算法的时候，要彻读论文，尤其是消融分析和补充材料（如果有）。
这些消减将使你直观地了解哪些参数或子例程对使事情正常运行具有最大的影响，这将帮助你诊断错误。
补充材料通常会提供有关特定细节的信息，例如网络体系结构和优化超参数，并且你应当尝试使实现与这些细节保持一致，以增加使它起作用的机会。

**但是不要过分解读论文的细节**。有时，论文使用了比严格必要更多的技巧，因此请对此有所警惕，并在可能的情况下尝试简化。
例如，原始DDPG论文提出了一种复杂的神经网络架构和初始化方案，以及批标准化。
这些并非绝对必要，并且DDPG的一些最佳报告结果都使用更简单的网络。
再举一个例子，原始的A3C论文使用了来自不同参与者学习者的异步更新，但是事实证明，同步更新也能正常工作。

**也不要过度解读现有的实现**。研究 `现有的`_ `实现`_ 以获取灵感，但请注意不要过度解读这些实现的工程细节。
强化学习库通常会做出抽象选择，这些选择有利于算法之间的代码重用，但是如果你仅编写单个算法或支持单个用例，则不必要。

**在简单的环境中快速迭代**。要调试你的实现，请在需要快速学习的简单环境中尝试它们，例如在 `OpenAI Gym`_ 的
CartPole-v0，InvertedPendulum-v0，FrozenLake-v0和HalfCheetah-v2上
（具有较短的时间范围──仅100或250步，而不是完整的1000步））。
如果你尚未首先验证算法可以在最简单的玩具任务上工作，请不要尝试在Atari或复杂的类人动物环境中运行算法。
在调试阶段，理想的实验周转时间小于5分钟（在本地计算机上），或者稍长一些，但不多。
这些小规模的实验不需要任何特殊的硬件，并且可以在CPU上运行而不会造成太多麻烦。

**如果不起作用，请假设存在错误**。在调整超参数之前，请花费大量精力查找错误：通常是错误。
错误的超参数可能会严重降低强化学习性能，但是如果你使用与论文和标准实施中的超参数相似的超参数，则可能不会成为问题。
还要牢记：有时即使在遇到严重错误的情况下，程序也可以在一种环境中运行，因此一旦结果令人满意，请确保在多个环境中进行测试。

**测量一切**。做很多测试以了解引擎盖下的情况。你在每次迭代中读取的学习过程的统计数据越多，调试起来就越容易──毕竟，如果看不到中断，就无法断定。
我个人希望查看累积奖励，剧集长度和价值函数估计值的均值/标准差/最小值/最大值，以及目标的损失以及任何探索参数的详细信息
（例如用于随机策略优化的均值熵，或当前的epsilon（如DQN中的epsilon-greedy）。
另外，不时观看有关智能体表现的视频；这将为你提供一些其他方面无法获得的见解。

**在工作正常时扩展实验规模**。在实现了强化学习算法，并且似乎可以在最简单的环境中正常运行的算法之后，请在较复杂的环境中对其进行测试。
此阶段的实验需要更长的时间，具体取决于几个小时到几天之间的某个时间。
此时，专用硬件（例如功能强大的GPU或32核机器）可能会有用，并且你应考虑研究AWS或GCE等云计算资源。

**保持这些习惯**！这些习惯值得你保持，不仅是在学习深度强化学习的阶段，它们将加速你的研究！


开展一个研究项目
=============================

一旦你对深度强化学习的基础知识感到相当满意，就应该开始突破界限并进行研究。要到达那里，你需要一个项目构想。

**从探索文献开始，以了解该领域的主题**。你可能会发现很多有趣的主题：
采样效率，探索，迁移学习，层次结构，内存，基于模型的强化学习，元学习和多智能体，仅举几例。
如果你正在寻找灵感，或者只是想大致了解其中的内容，
请查看Spinning Up的 `关键论文列表 <../spinningup/keypapers.html>`_。
找到你喜欢的其中一个主题的论文（启发你的东西）并仔细阅读。
使用相关的工作部分和引用来查找密切相关的论文，并深入研究文献。
你将开始找出未解决的问题以及可以在何处你可以产生影响。

**产生想法的方法**。有很多不同的方法可以开始考虑项目的想法，并且你选择的框架会影响项目的发展方式以及面临的风险。这里有一些例子：

**方法1：改进现有方法**。这是增量主义的角度，你可以通过调整现有算法来尝试在已建立的问题设置中获得性能提升。
在这里重新实现先前的工作非常有帮助，因为它使你了解现有算法脆弱且可以改进的方式。
新手会发现这是最容易使用的方法，但是对于任何有经验的研究人员来说，它也是值得的。
尽管一些研究人员发现增量主义并不那么令人兴奋，但机器学习中一些最令人印象深刻的成就却来自这种性质的工作。

由于此类项目与现有方法相关，因此它们的范围很窄，可以很快（几个月）结束，这可能是理想的（特别是作为研究人员起步时）。
但这也带来了风险：你可能对算法进行的调整可能无法改善它，在这种情况下，除非你提出更多的调整，否则项目就结束了，并且你还没有明确的信号接下来做什么。

.**方法2：关注未解决的基准**。而不是考虑如何改进现有方法，你的目标是成功完成一项以前没有人解决过的任务。
例如：在Sonic领域或Gym Retro中实现从训练水平到测试水平的完美泛化。
当你完成一项尚未解决的任务时，你可能会尝试多种方法，包括先前的方法和为该项目发明的新方法。
新手可以解决此类问题，但学习曲线会更加陡峭。

此方法中的项目范围很广，可以持续一段时间（几个月到一年以上）。
主要风险在于，如果没有重大突破，基准是无法解决的，这意味着很容易花费大量时间而没有取得任何进展。
但是，即使像这样的项目失败了，它也通常会导致研究人员获得许多新见解，这些见解将成为下一个项目的沃土。

**方法3：创建新的问题设定**。与其考虑现有方法或当前面临的巨大挑战，不如考虑一个尚未研究的完全不同的概念性问题。
然后，找出如何取得进展。对于遵循这些思路的项目，可能尚不存在标准基准，因此你必须设计一个基准。
这可能是一个巨大的挑战，但值得拥抱──出色的基准可以使整个领域向前发展。

问题出现时就会出现此框架中的问题──很难去寻找它们。

**避免重新发明轮子**。当你想到要开始测试的好主意时，那就太好了！
但是，尽管你仍处于初期阶段，但请进行最彻底的检查，以确保尚未完成。
半途而废地完成一个项目，然后才发现已经有关于你的想法的论文可能会令人沮丧。
当工作同时进行时，这尤其令人沮丧，这不时发生！
但是，不要让那吓到你──绝对不要让它激励你以尚未完成的研究来打下烙印，并夸大了部分工作的优点。
做一个好的研究，并通过彻底而彻底的调查来完成你的项目，因为从长远来看，这才是最重要的。


做严谨的强化学习研究
=============================

The claim you'll make in your work is that those design decisions collectively help, but this is really a bundle of several claims in disguise: one for each such design element. 

现在你想出了一个主意，并且可以肯定它还没有完成。你将使用开发的技能来实施它，然后开始在标准域上对其进行测试。看起来可行！
但是，这意味着什么？它必须发挥多大的作用才能变得重要？这是深度强化学习研究中最难的部分之一。
为了验证你的建议是有意义的贡献，你必须严格证明它实际上比最强大的基准算法（在当前在测试域上达到SOTA（最新技术水平））取得了性能上的好处。
如果你发明了一个新的测试域，那么就没有以前的SOTA了，你仍然需要尝试一下文献中最可靠的算法，该算法在新的测试域中可能表现良好，然后就必须击败它。

**进行公平比较**。如果你是从头开始实施基准线（而不是直接与另一张纸的数字进行比较），那么花在调整基准线上的时间与调整自己的算法所花的时间一样重要。
这将确保比较是公平的。此外，即使算法与基准之间存在实质性差异，也应尽力保持“其他条件不变”。
例如，如果你要研究架构变体，则使模型参数的数量在模型和基准之间大致相等。在任何情况下均不得妨碍基线！
事实证明，强化学习中的基准非常强大，要想获得更大，持续的胜利可能会很棘手，或者需要对算法设计有一定的了解。

**消除作为混杂因素的随机性**。当心随机种子会使事情看起来比实际更强或更弱，因此请为许多随机种子运行所有命令（至少3个，但如果要透彻，则执行10个或更多）。
这确实很重要，值得重点关注：在许多常见的用例中，深度强化学习相对于随机种子而言似乎很脆弱。
可能存在足够的方差，两组不同的随机种子可以产生差异很大的学习曲线，以至于它们看起来根本不是来自同一分布（请参见 `此处的图10`_）。

**运行高完整性实验**。不要仅仅从最佳或最有趣的运行中获取结果以用于你的论文中。
相反，针对你打算比较的所有方法（如果要与自己的基准实现进行比较）启动新的最终实验，并预先承诺报告其中的结果。
这是为了强制执行一种较弱的 `预注册形式`_：你使用调整阶段来得出你的假设，然后使用最终运行来得出你的结论。

**分别检查每个设计选择**。做研究的另一个关键方面是进行消融分析。
你提出的任何方法都可能具有多个关键设计决策，例如体系结构选择或正则化技术，每个决策都可能分别影响性能。
你将在工作中提出的主张是这些设计选择可以共同帮助，但这实际上是一堆变相的声明：每个这样的设计元素。
通过系统地评估如果将其替换为其他设计选择或将其完全删除会发生什么，你可以弄清楚如何正确地归功于你的方法所带来的好处。
这样一来，你就可以对每个单独的设计选择充满信心，并提高工作的整体实力。


别想太多
================

Deep RL是一个令人兴奋，快速发展的领域，我们需要尽可能多的人来解决开放的问题并在这些问题上取得进展。
希望你在阅读本文后感到更加准备加入其中！当你准备就绪时，`请告诉我们`_。

.. _`请告诉我们`: https://jobs.lever.co/openai


以及：其他资源
===================

考虑阅读以下其他有关在该领域成为研究人员或工程师的内容丰富的文章：

`短期机器学习研究项目的建议 <https://rockt.github.io/2018/08/29/msc-advice>`_，Tim Rocktäschel，Jakob Foerster 和 Greg Farquhar。

`用于AI安全性和健壮性的机器学习工程化：《来自谷歌大脑工程师的指南》 <https://80000hours.org/articles/ml-engineering-career-transition-guide/>`_，Catherine Olsson 和 80,000 小时。


参考
==========

.. _`它很难不总是有效`: https://www.alexirpan.com/2018/02/14/rl-hard.html
.. [1] `Deep Reinforcement Learning Doesn't Work Yet <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_, Alex Irpan, 2018

.. _`可重现性`: https://arxiv.org/abs/1708.04133
.. _`此处的图10`: https://arxiv.org/pdf/1708.04133.pdf
.. [2] `Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control <https://arxiv.org/abs/1708.04133>`_, Islam et al, 2017

.. _`一大挑战`: https://arxiv.org/abs/1709.06560
.. [3] `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_, Henderson et al, 2017

.. _`学习的曲线非常陡峭`: http://amid.fish/reproducing-deep-rl
.. [4] `Lessons Learned Reproducing a Deep Reinforcement Learning Paper <http://amid.fish/reproducing-deep-rl>`_, Matthew Rahtz, 2018

.. _`很多`: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
.. [5] `UCL Course on RL <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>`_

.. _`很棒的`: http://rll.berkeley.edu/deeprlcourse/
.. [6] `Berkeley Deep RL Course <http://rll.berkeley.edu/deeprlcourse/>`_

.. _`学习`: https://sites.google.com/view/deep-rl-bootcamp/lectures
.. [7] `Deep RL Bootcamp <https://sites.google.com/view/deep-rl-bootcamp/lectures>`_

.. _`资源`: http://joschu.net/docs/nuts-and-bolts.pdf
.. [8] `Nuts and Bolts of Deep RL <http://joschu.net/docs/nuts-and-bolts.pdf>`_, John Schulman

.. _`多层感知机`: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
.. [9] `Stanford Deep Learning Tutorial: Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_

.. _`Vanilla RNN`: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. [10] `The Unreasonable Effectiveness of Recurrent Neural Networks <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_, Andrej Karpathy, 2015

.. _`LSTM`: https://arxiv.org/abs/1503.04069
.. [11] `LSTM: A Search Space Odyssey <https://arxiv.org/abs/1503.04069>`_, Greff et al, 2015

.. _`同时看这篇博客`: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. [12] `Understanding LSTM Networks <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_, Chris Olah, 2015

.. _`GRU`: https://arxiv.org/abs/1412.3555v1
.. [13] `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555v1>`_, Chung et al, 2014 (GRU paper)

.. _`卷积`: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
.. [14] `Conv Nets: A Modular Perspective <http://colah.github.io/posts/2014-07-Conv-Nets-Modular/>`_, Chris Olah, 2014

.. _`层`: https://cs231n.github.io/convolutional-networks/
.. [15] `Stanford CS231n, Convolutional Neural Networks for Visual Recognition <https://cs231n.github.io/convolutional-networks/>`_

.. _`残差网络`: https://arxiv.org/abs/1512.03385
.. [16] `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_, He et al, 2015 (ResNets)

.. _`注意力`: https://arxiv.org/abs/1409.0473
.. [17] `Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473>`_, Bahdanau et al, 2014 (Attention mechanisms)

.. _`机制`: https://arxiv.org/abs/1706.03762
.. [18] `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_, Vaswani et al, 2017

.. _`weight decay`: https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf
.. [19] `A Simple Weight Decay Can Improve Generalization <https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf>`_, Krogh and Hertz, 1992


.. _`dropout`: http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
.. [20] `Dropout:  A Simple Way to Prevent Neural Networks from Overfitting <http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf>`_, Srivastava et al, 2014

.. _`batch norm`: https://arxiv.org/abs/1502.03167
.. [21] `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_, Ioffe and Szegedy, 2015

.. _`layer norm`: https://arxiv.org/abs/1607.06450
.. [22] `Layer Normalization <https://arxiv.org/abs/1607.06450>`_, Ba et al, 2016

.. _`weight norm`: https://arxiv.org/abs/1602.07868
.. [23] `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/abs/1602.07868>`_, Salimans and Kingma, 2016

.. _`SGD, momentum SGD`: http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
.. [24] `Stanford Deep Learning Tutorial: Stochastic Gradient Descent <http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/>`_

.. _`Adam`: https://arxiv.org/abs/1412.6980
.. [25] `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_, Kingma and Ba, 2014

.. _`以及其它`: https://arxiv.org/abs/1609.04747
.. [26] `An overview of gradient descent optimization algorithms <https://arxiv.org/abs/1609.04747>`_, Sebastian Ruder, 2016

.. _`reparameterization trick`: https://arxiv.org/abs/1312.6114
.. [27] `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_, Kingma and Welling, 2013 (Reparameterization trick)

.. _`Tensorflow`: https://www.tensorflow.org/
.. [28] `Tensorflow`_

.. _`PyTorch`: http://pytorch.org/
.. [29] `PyTorch`_

.. _`介绍`: ../spinningup/rl_intro.html
.. [30] `Spinning Up强化学习，第一部分：强化学习中的核心概念 <../spinningup/rl_intro.html>`_

.. _`强化学习介绍`: https://github.com/jachiam/rl-intro/blob/master/Presentation/rl_intro.pdf
.. [31] `强化学习介绍`_ Slides from OpenAI Hackathon, Josh Achiam, 2018

.. _`Lilian Weng 的综述`: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
.. [32] `A (Long) Peek into Reinforcement Learning <https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>`_, Lilian Weng, 2018

.. _`单调提升理论`: http://joschu.net/docs/thesis.pdf
.. [33] `Optimizing Expectations <http://joschu.net/docs/thesis.pdf>`_, John Schulman, 2016 (Monotonic improvement theory)

.. _`经典强化学习算法`: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
.. [34] `Algorithms for Reinforcement Learning <https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf>`_, Csaba Szepesvari, 2009 (Classic RL Algorithms)

.. _`REINFORCE`: https://arxiv.org/abs/1604.06778
.. [35] `Benchmarking Deep Reinforcement Learning for Continuous Control <https://arxiv.org/abs/1604.06778>`_, Duan et al, 2016

.. _`DQN`: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
.. [36] `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_, Mnih et al, 2013 (DQN)

.. _`A2C`: https://blog.openai.com/baselines-acktr-a2c/
.. [37] `OpenAI Baselines: ACKTR & A2C <https://blog.openai.com/baselines-acktr-a2c/>`_

.. _`A3C`: https://arxiv.org/abs/1602.01783
.. [38] `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_, Mnih et al, 2016 (A3C)

.. _`PPO`: https://arxiv.org/abs/1707.06347
.. [39] `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_, Schulman et al, 2017 (PPO)

.. _`DDPG`: https://arxiv.org/abs/1509.02971
.. [40] `Continuous Control with Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_, Lillicrap et al, 2015 (DDPG)

.. _`简洁版本的VPG`: https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py
.. [41] `RL-Intro Policy Gradient Sample Code <https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py>`_, Josh Achiam, 2018

.. _`现有的`: https://github.com/openai/baselines
.. [42] `OpenAI Baselines <https://github.com/openai/baselines>`_

.. _`实现`: https://github.com/rll/rllab
.. [43] `rllab <https://github.com/rll/rllab>`_

.. _`OpenAI Gym`: https://gym.openai.com/
.. [44] `OpenAI Gym <https://gym.openai.com/>`_

.. _`Sonic domain`: https://contest.openai.com/2018-1/
.. [45] `OpenAI Retro Contest <https://contest.openai.com/2018-1/>`_

.. _`Gym Retro`: https://blog.openai.com/gym-retro/
.. [46] `OpenAI Gym Retro <https://blog.openai.com/gym-retro/>`_

.. _`预注册形式`: https://cos.io/prereg/
.. [47] `Center for Open Science <https://cos.io/prereg/>`_, explaining what preregistration means in the context of scientific experiments.
