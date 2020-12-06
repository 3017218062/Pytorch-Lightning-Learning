[TOC]

# Callback

## 简介

Pytorch Lightning最nb的插件，万能，无敌，随处可插，即插即用。

## 方法

### 训练方法

#### on_train_start(trainer, pl_module)

当第一次训练开始时的操作。

#### on_train_end(trainer, pl_module)

当最后一次训练结束时的操作。

#### on_train_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

当一批数据训练开始时的操作。

#### on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

当一批数据训练结束时的操作。

#### on_train_epoch_start(trainer, pl_module)

当一轮数据训练开始时的操作。

#### on_train_epoch_end(trainer, pl_module, outputs)

当一轮数据训练结束时的操作。

### 验证方法

#### on_validation_start(trainer, pl_module)

当第一次验证开始时的操作。

#### on_validation_end(self, trainer, pl_module)

当最后一次验证结束时的操作。

#### on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

当一批数据验证开始时的操作。

#### on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

当一批数据验证结束时的操作。

#### on_validation_epoch_start(trainer, pl_module)

当一轮数据验证开始时的操作。

#### on_validation_epoch_end(trainer, pl_module)

当一轮数据验证结束时的操作。

### 测试方法

#### on_test_start(trainer, pl_module)

当第一次测试开始时的操作。

#### on_test_end(self, trainer, pl_module)

当最后一次测试结束时的操作。

#### on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

当一批数据测试开始时的操作。

#### on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

当一批数据测试结束时的操作。

#### on_test_epoch_start(trainer, pl_module)

当一轮数据测试开始时的操作。

#### on_test_epoch_end(trainer, pl_module)

当一轮数据测试结束时的操作。

### 其他方法

#### on_fit_start(trainer, pl_module)

当调用.fit时的操作。

#### on_fit_end(trainer, pl_module)

.fit结束时的操作。

#### setup(trainer, pl_module, stage)

#### teardown(trainer, pl_module, stage)

#### on_init_start(trainer)

#### on_init_end(trainer)

#### on_sanity_check_start(trainer, pl_module)

#### on_sanity_check_end(trainer, pl_module)

#### on_batch_start(trainer, pl_module)

#### on_batch_end(trainer, pl_module)

#### on_epoch_start(trainer, pl_module)

#### on_epoch_end(trainer, pl_module)

#### on_keyboard_interrupt(trainer, pl_module)

#### on_save_checkpoint(trainer, pl_module)

#### on_load_checkpoint(checkpointed_state)

# 哪些需要掌握，哪些不需要

类似于LightningModule的各种方法，需要进行操作时在对应位置进行修改。后面将会举几个实用的例子。