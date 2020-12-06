[TOC]

# Trainer

## 简介

Pytorch Lightning的两大API之一，类似于“胶水”，将LightningModule各个部分连接形成完整的逻辑。

## 方法

#### \_\_init\_\_(logger=True, checkpoint_callback=True, callbacks=None, default_root_dir=None, gradient_clip_val=0, process_position=0, num_nodes=1, num_processes=1, gpus=None, auto_select_gpus=False, tpu_cores=None, log_gpu_memory=None, progress_bar_refresh_rate=1, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=1, max_epochs=1000, min_epochs=1, max_steps=None, min_steps=None, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, val_check_interval=1.0, flush_logs_every_n_steps=100, log_every_n_steps=50, accelerator=None, sync_batchnorm=False, precision=32, weights_summary='top', weights_save_path=None, num_sanity_val_steps=2, truncated_bptt_steps=None, resume_from_checkpoint=None, profiler=None, benchmark=False, deterministic=False, reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=True, terminate_on_nan=False, auto_scale_batch_size=False, prepare_data_per_node=True, plugins=None, amp_backend='native', amp_level='O2', distributed_backend=None, automatic_optimization=True, move_metrics_to_cpu=False)

初始化训练器，参数很多，下面将分别介绍：

- 硬件参数：
  - gpus[None]:
    - 设置为0或None，表示使用cpu。
    - 设置为大于0的整数n，表示使用n块gpu。
    - 设置为大于0的整数字符串'n'，表示使用id为n的gpu。
    - 设置为-1或'-1'，表示使用所有gpu。
    - 设置为整数数组[a, b]或整数数组字符串'a, b'，表示使用id为a和b的gpu。
  - auto_select_gpus[False]:
    - 设置为True，自动选择所需gpu。
    - 设置为False，按顺序选择所需gpu。
  - num_nodes[1]:
    - 设置为1，选择当前gpu节点。
    - 设置为大于0的整数n，表示使用n个节点。
  - tpu_cores[None]:
    - 设置为None，表示不使用tpu。
    - 设置为1，表示使用1个tpu内核。
    - 设置为大于0的整数数组[n]，表示使用id为n的tpu内核。
    - 设置为8，表示使用所有tpu内核。
- 精度参数：
  - precision[32]:
    - 设置为2、4、8、16或32，分别表示不同的精度。
  - amp_backend["native"]:
    - 设置为"native"，表示使用本地混合精度。
    - 设置为"apex"，表示使用apex混合精度。
  - amp_level["O2"]:
    - 设置为O0、O1、O2或O3，分别表示:
      - O0：纯FP32训练，可以作为accuracy的baseline。
      - O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
      - O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
      - O3：纯FP16训练，很不稳定，但是可以作为speed的baseline。
- 训练超参：
  - max_epochs[1000]:
    - 最大训练轮数。
  - min_epochs[1]:
    - 最小训练轮数。
  - max_steps[None]:
    - 每轮最大训练步数。
  - min_steps[None]:
    - 每轮最小训练步数。
- 日志参数和检查点参数：
  - checkpoint_callback[True]:
    - 设置为True，自动进行检查点保存。
    - 设置为False，不进行检查点保存。
  - logger[TensorBoardLogger]:
    - 设置log工具。False表示不使用logger。
  - default_root_dir[os.getcwd()]:
    - 默认的根目录，用于日志和检查点的保存。
  - flush_logs_every_n_steps[100]:
    - 多少步更新一次日志到磁盘。
  - log_every_n_steps[50]:
    - 多少步更新一次日志到内存。
  - log_gpu_memory[None]:
    - 设置为None，不记录gpu显存信息。
    - 设置为"all"，记录所有gpu显存信息。
    - 设置为"min_max"，记录gpu显存信息最值。
  - check_val_every_n_epoch[1]:
    - 多少轮验证一次。
  - val_check_interval[1.0]:
    - 设置为小数，表示取一定比例的验证集。
    - 设置为整数，表示取一定数量的验证集。
  - resume_from_checkpoint[None]:
    - 检查点恢复，输入路径。
  - progress_bar_refresh_rate[1]:
    - 进度条的刷新率。
  - weights_summary["top"]:
    - 设置为None，不输出模型信息。
    - 设置为"top"，输出模型简要信息。
    - 设置为"full"，输出模型所有信息。
  - weights_save_path[os.getcwd()]:
    - 权重的保存路径。
- 测试参数：
  - num_sanity_val_steps[2]:
    - 训练前检查多少批验证数据。
  - fast_dev_run[False]:
    - 一系列单元测试。
  - reload_dataloaders_every_epoch[False]:
    - 每一轮是否重新载入数据。
- 分布式参数：
  - accelerator[None]:
    - dp（DataParallel）是在同一计算机的GPU之间拆分批处理。
    - ddp（DistributedDataParallel）是每个节点上的每个GPU训练并同步梯度。TPU默认选项。

    - ddp_cpu（CPU上的DistributedDataParallel）与ddp相同，但不使用GPU。对于多节点CPU训练或单节点调试很有用。

    - ddp2是节点上的dp，节点间的ddp。
  - accumulate_grad_batches[1]:
    - 多少批进行一次梯度累积。
  - sync_batchnorm[False]:
    - 同步批处理，一般是在分布式多GPU时使用。
- 自动参数：
  - automatic_optimization[True]:
    - 是否开启自动优化。
  - auto_scale_batch_size[None]:
    - 是否自动寻找最大批大小。
  - auto_lr_find[False]:
    - 是否自动寻找最佳学习率。
- 确定性参数：
  - benchmark[False]:
    - 是否使用cudnn.benchmark。
  - deterministic[False]:
    - 是否开启确定性。
- 限制性参数和采样参数：
  - gradient_clip_val[0.0]:
    - 梯度裁剪。
  - limit_train_batches[1.0]:
    - 限制每轮的训练批次数量。
  - limit_val_batches[1.0]:
    - 限制每轮的验证批次数量。
  - limit_test_batches[1.0]:
    - 限制每轮的测试批次数量。
  - overfit_batches[0.0]:
    - 限制批次的重复数量。
  - prepare_data_per_node[True]:
    - 是否对每个结点准备数据。
  - replace_sampler_ddp[True]:
    - 是否启用自动添加分布式采样器的功能。
- 其他参数：
  - callbacks[]:
    - 好家伙，callback。
  - process_position[0]:
    - 对进度条进行有序处理。
  - profiler[None]
  - track_grad_norm[-1]
  - truncated_bptt_steps[None]

#### fit(model, train_dataloader=None, val_dataloaders=None, datamodule=None)

开启训练。参数如下：

- datamodule (Optional[LightningDataModule]) – 一个LightningDataModule实例。
- model (LightningModule) – 训练的模型。
- train_dataloader (Optional[DataLoader]) – 训练数据。
- val_dataloaders (Union[DataLoader, List[DataLoader], None]) – 验证数据。

#### test(model=None, test_dataloaders=None, ckpt_path='best', verbose=True, datamodule=None)

开启测试。参数如下：

- ckpt_path (Optional[str]) – best或者你最希望测试的检查点权重的路径，None使用最后的权重。
- datamodule (Optional[LightningDataModule]) – 一个LightningDataModule实例。
- model (Optional[LightningModule]) – 测试的模型。
- test_dataloaders (Union[DataLoader, List[DataLoader], None]) –  测试数据。
- verbose (bool) – 是否打印结果。

#### tune(model, train_dataloader=None, val_dataloaders=None, datamodule=None)

训练之前调整超参数。参数如下：

- datamodule (Optional[LightningDataModule]) – 一个LightningDataModule实例。
- model (LightningModule) – 调整的模型。
- train_dataloader (Optional[DataLoader]) – 训练数据。
- val_dataloaders (Union[DataLoader, List[DataLoader], None]) – 验证数据。

## 属性

#### callback_metrics

回调指标。

举个例子：

```python
def training_step(self, batch, batch_idx):
    self.log('a_val', 2)

callback_metrics = trainer.callback_metricpythons
assert callback_metrics['a_val'] == 2
```

#### current_epoch

当前轮数。

举个例子：

```python
def training_step(self, batch, batch_idx):
    current_epoch = self.trainer.current_epoch
    if current_epoch > 100:
        # do something
        pass
```

#### logger

当前日志。

举个例子：

```python
def training_step(self, batch, batch_idx):
    logger = self.trainer.logger
    tensorboard = logger.experiment
```

#### logged_metrics

发送到日志的指标。

举个例子：

```python
def training_step(self, batch, batch_idx):
    self.log('a_val', 2, log=True)

logged_metrics = trainer.logged_metrics
assert logged_metrics['a_val'] == 2
```

#### log_dir

当前目录，用于保存图像等。

举个例子：

```python
def training_step(self, batch, batch_idx):
    img = ...
    save_img(img, self.trainer.log_dir)
```

#### is_global_zero

是否为全局第一个。

#### progress_bar_metrics

发送到进度条的指标。

举个例子：

```python
def training_step(self, batch, batch_idx):
    self.log('a_val', 2, prog_bar=True)

progress_bar_metrics = trainer.progress_bar_metrics
assert progress_bar_metrics['a_val'] == 2
```

# 哪些需要掌握，哪些不需要

- 需要掌握的方法：
  - \_\_init\_\_（参数比较多，可以花时间记录一下自己最常用的参数配置）
  - fit/tune
- 需要掌握的属性：
  - current_epoch

其他的需要时再看即可。