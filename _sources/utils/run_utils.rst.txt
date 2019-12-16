=========
运行工具
=========

.. contents:: 目录


ExperimentGrid
==============

Spinning Up提供了一个叫ExperimentGrid的工具，可简化超参数。
这是基于 `rllab工具`_ 的（但比它简单），该工具名为VariantGenerator。

.. _`rllab工具`: https://github.com/rll/rllab/blob/master/rllab/misc/instrument.py#L173

.. autoclass:: spinup.utils.run_utils.ExperimentGrid
    :members:


运行实验
===================

.. autofunction:: spinup.utils.run_utils.call_experiment

.. autofunction:: spinup.utils.run_utils.setup_logger_kwargs
