[TOC]

# LightningModule

## 简介

Pytorch Lightning的两大API之一，是torch.nn.Module的高级封装。

## 方法

### 定义模型

#### \_\_init\_\_()

同torch.nn.Module中的\_\_init\_\_，用于构建模型。

#### forward(\*args, \*\*kwargs)

同torch.nn.Module中的forward，通过\_\_init\_\_中的各个模块实现前向传播。

### 训练模型

#### training_step(\*args, \*\*kwargs)

训练一批数据并反向传播。参数如下：

- batch (Tensor | (Tensor, …) | [Tensor, …]) – 数据输入，一般为x, y = batch。
- batch_idx (int) – 批次索引。
- optimizer_idx (int) – 当使用多个优化器时，会使用本参数。
- hiddens (Tensor) – 当truncated_bptt_steps > 0时使用。

举个例子：

```python
def training_step(self, batch, batch_idx): # 数据类型自动转换，模型自动调用.train()
    x, y = batch
    _y = self(x)
    loss = criterion(_y, y) # 计算loss
    return loss # 返回loss，更新网络

def training_step(self, batch, batch_idx, hiddens):
    # hiddens是上一次截断反向传播的隐藏状态
    out, hiddens = self.lstm(data, hiddens)
    return {"loss": loss, "hiddens": hiddens}
```

#### training_step_end(\*args, \*\*kwargs)

一批数据训练结束时的操作。一般用不着，分布式训练的时候会用上。参数如下：

- batch_parts_outputs – 当前批次的training_step()的返回值

举个例子：

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    _y = self(x)
    return {"output": _y， "target": y}

def training_step_end(self, training_step_outputs): # 多GPU分布式训练,计算loss
    gpu_0_output = training_step_outputs[0]["output"]
    gpu_1_output = training_step_outputs[1]["output"]
    
    gpu_0_target = training_step_outputs[0]["target"]
    gpu_1_target = training_step_outputs[1]["target"]

    # 对所有GPU的数据进行处理
    loss = criterion([gpu_0_output, gpu_1_output]， [gpu_0_target, gpu_1_target])
    return loss
```

#### training_epoch_end(outputs)

一轮数据训练结束时的操作。主要针对于本轮所有training_step的输出。参数如下：

- outputs (List[Any]) – training_step()的输出。

举个例子：

```python
def training_epoch_end(self, outs): # 计算本轮的loss和acc
    loss = 0.
    for out in outs: # outs按照训练顺序排序
        loss += out["loss"].cpu().detach().item()
    loss /= len(outs)
    acc = self.train_metric.compute()

    self.history["loss"].append(loss)
    self.history["acc"].append(acc)
```

### 验证模型

#### validation_step(\*args, \*\*kwargs)

见training_step。

#### validation_step_end(\*args, \*\*kwargs)

见training_step_end。

#### validation_epoch_end(outputs)

见training_epoch_end。

### 测试模型

#### test_step(\*args, \*\*kwargs)

见training_step。

#### test_step_end(\*args, \*\*kwargs)

见training_step_end。

#### test_epoch_end(outputs)

见training_epoch_end。

### 其他有用的功能

#### configure_optimizers()

在优化过程中选择优化器和学习率调度器，通常只需要一个，但对于GAN之类的可能需要多个。

举一堆例子：

- 单个优化器

  ```python
  def configure_optimizers(self):
      return Adam(self.parameters(), lr=1e-3)
  ```

- 多个优化器（比如GAN）

  ```python
  def configure_optimizers(self):
      generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
      disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
      return generator_opt, disriminator_opt
  ```

  可以修改frequency键来控制优化频率：

  ```python
  def configure_optimizers(self):
      gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
      dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
      n_critic = 5
      return (
          {"optimizer": dis_opt, "frequency": n_critic},
          {"optimizer": gen_opt, "frequency": 1}
      )
  ```

- 多个优化器和多个调度器或学习率字典（比如GAN）

  ```python
  def configure_optimizers(self):
      generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
      disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
      discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
      return [generator_opt, disriminator_opt], [discriminator_sched]
  ```

  ```python
  def configure_optimizers(self):
      generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
      disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
      discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
      return {"optimizer": [generator_opt, disriminator_opt], "lr_scheduler": [discriminator_sched]}
  ```

  对于调度器，可以修改其属性：

  ```python
  {
      "scheduler": lr_scheduler, # 调度器
      "interval": "epoch", # 调度的单位，epoch或step
      "frequency": 1, # 调度的频率，多少轮一次
      "reduce_on_plateau": False, # ReduceLROnPlateau
      "monitor": "val_loss", # ReduceLROnPlateau的监控指标
      "strict": True # 如果没有monitor，是否中断训练
  }
  ```

  ```python
  def configure_optimizers(self):
      gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
      dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
      gen_sched = {"scheduler": ExponentialLR(gen_opt, 0.99), "interval": "step"}
      dis_sched = CosineAnnealing(discriminator_opt, T_max=10)
      return [gen_opt, dis_opt], [gen_sched, dis_sched]
  ```

一些注意事项：

- Lightning在需要的时候会调用backward和step。

- 如果使用半精度（precision=16），Lightning会自动处理。

- 如果使用多个优化器，training_step会附加一个参数optimizer_idx。

- 如果使用LBFGS，Lightning将自动为您处理关闭功能。

- 如果使用多个优化器，则在每个训练步骤中仅针对当前优化器的参数计算梯度。

- 如果您需要控制这些优化程序执行或改写默认step的频率，请改写optimizer_step。

- 如果在每n步都调用调度器，或者只想监视自定义指标，则可以在lr_dict中指定它们。

  ```python
  {
      "scheduler": lr_scheduler,
      "interval": "step",  # or "epoch"
      "monitor": "val_f1",
      "frequency": n,
  }
  ```

#### freeze()/unfreeze()

冻结所有参数/解冻所有参数。

#### save_hyperparameters(\*args, frame=None)

保存\_\_init\_\_中传入的超参数。

举个例子：

```python
def __init__(self, arg1, arg2, arg3): # 1, "abc", 3.14
	super().__init__()
    # self.save_hyperparameters() # 保存所有超参数
    # self.save_hyperparameters("arg1", "arg2", "arg3") # 同上，保存所有超参数
    self.save_hyperparameters("arg1", "arg3") # 保存部分超参数

def __init__(self, params): # params=Namespace(p1=1, p2="abc", p3=3.14)
	super().__init__()
    self.save_hyperparameters(params) # 保存所有超参数
```

#### to_onnx(file_path, input_sample=None, \*\*kwargs)

保存模型为ONNX格式。参数如下：

- file_path (str) – 保存路径。
- input_sample (Optional[Tensor]) – 用于跟踪的输入张量的样本。
- **kwargs – 将传递给torch.onnx.export函数。

举个例子：

```python
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
    model = SimpleModel()
    input_sample = torch.randn((1, 64))
    model.to_onnx(tmpfile.name, input_sample, export_params=True)
    os.path.isfile(tmpfile.name)
```

### 其他没啥用的功能

#### log(name, value, prog_bar=False, logger=True, on_step=None, on_epoch=None, reduce_fx=torch.mean, tbptt_reduce_fx=torch.mean, tbptt_pad_token=0, enable_graph=False, sync_dist=False, sync_dist_op="mean", sync_dist_group=None)

#### log_dict(dictionary, prog_bar=False, logger=True, on_step=None, on_epoch=None, reduce_fx=torch.mean, tbptt_reduce_fx=torch.mean, tbptt_pad_token=0, enable_graph=False, sync_dist=False, sync_dist_op="mean", sync_dist_group=None)

#### print(\*args, \*\*kwargs)

#### to_torchscript(file_path=None, method=\'script\', example_inputs=None, \*\*kwargs)

有人问，log不重要吗？当然重要，但不是用Pytorch Lightning的log，因为实在是太反人类了，各种bug，后续我们会用LightningModule和Callback实现我们自己的logger。

## 属性

#### current_epoch

当前轮数。

#### device

当前设备。

#### global_rank

全局排名是指该GPU在所有GPU中的索引。如果使用10台计算机，每台计算机具有4个GPU，则第10台计算机上的第4个GPU的global_rank = 39。

Lightning仅从开始global_rank = 0保存日志，权重等。通常不需要使用此属性。

#### global_step

当前步数，每轮不重置。

#### hparams

save_hyperparameters所保存的超参数。

#### logger（不推荐使用）

当前日志。

#### local_rank

本地排名是指该计算机上的索引。如果使用10台计算机，则每台计算机上索引为0的GPU的local_rank = 0。

Lightning仅从开始global_rank = 0保存日志，权重等。通常不需要使用此属性。

#### precision

所使用的进度类型。

#### trainer

指向trainer。

#### use_amp/use_ddp/use_ddp2/use_dp/use_tpu

是否使用了自动混合精度/ddp/ddp2/dp/tpu。

## 更多自定义方法

下面的伪代码描述了训练过程：

```python
def fit(...):
    on_fit_start()

    if global_rank == 0:
        # prepare data is called on GLOBAL_ZERO only
        prepare_data()

    for gpu/tpu in gpu/tpus:
        train_on_device(model.copy())

    on_fit_end()

def train_on_device(model):
    # setup is called PER DEVICE
    setup()
    configure_optimizers()
    on_pretrain_routine_start()

    for epoch in epochs:
        train_loop()

    teardown()

def train_loop():
    on_train_epoch_start()
    train_outs = []
    for train_batch in train_dataloader():
        on_train_batch_start()

        # ----- train_step methods -------
        out = training_step(batch)
        train_outs.append(out)

        loss = out.loss

        backward()
        on_after_backward()
        optimizer_step()
        on_before_zero_grad()
        optimizer_zero_grad()

        on_train_batch_end(out)

        if should_check_val:
            val_loop()

    # end training epoch
    logs = training_epoch_end(outs)

def val_loop():
    model.eval()
    torch.set_grad_enabled(False)

    on_validation_epoch_start()
    val_outs = []
    for val_batch in val_dataloader():
        on_validation_batch_start()

        # -------- val step methods -------
        out = validation_step(val_batch)
        val_outs.append(out)

        on_validation_batch_end(out)

    validation_epoch_end(val_outs)
    on_validation_epoch_end()

    # set up for train
    model.train()
    torch.set_grad_enabled(True)
```

#### backward(loss, optimizer, optimizer_idx, \*args, \*\*kwargs)

反向传播。参数如下：

- loss (Tensor) – 已经被积累的梯度所放缩的损失。
- optimizer (Optimizer) – 当前被使用的优化器。
- optimizer_idx (int) – 当前被使用的优化器的索引。

举个例子：

```python
def backward(self, loss, optimizer, optimizer_idx):
    loss.backward()
```

#### get_progress_bar_dict()

修改进度条的内容。

举个例子：

```python
# Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]

def get_progress_bar_dict(self):
    # 不显示v_num
    items = super().get_progress_bar_dict()
    items.pop("v_num", None)
    return items
```

#### manual_backward(loss, optimizer, \*args, \*\*kwargs)

手动反向传播。无法覆盖trainer的设置。

举个例子：

```python
def training_step(...):
    (opt_a, opt_b) = self.optimizers()
    loss = ...
    self.manual_backward(loss, opt_a)
    self.manual_optimizer_step(opt_a)
```

#### manual_optimizer_step(optimizer, force_optimizer_step=False)

手动优化。无法覆盖trainer的设置。参数如下：

- optimizer (Optimizer) – 用于step()的优化器。
- force_optimizer_step (bool) – 是否强制执行优化程序步骤。当有2个优化器且其中一个应使用累积的渐变而不是另一个使用渐变时，这可能会很有用。可以采用自己的逻辑来强制执行优化程序步骤。

举个例子：

```python
def training_step(...):
    (opt_a, opt_b) = self.optimizers()
    loss = ...
    self.manual_backward(loss, opt_a)
    self.manual_optimizer_step(opt_a, force_optimizer_step=True)
```

#### on_after_backward()

在loss.backward()之后且优化程序执行任何操作之前在训练循环中调用。这是检查或记录梯度信息的理想位置。

举个例子：

```python
def on_after_backward(self):
    if self.trainer.global_step % 25 == 0:
        params = self.state_dict()
        for k, v in params.items():
            grads = v
            name = k
            self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)
```

#### on_before_zero_grad(optimizer)

在optimizer.step()之后和optimizer.zero_grad()之前调用。检查权重信息并更新权重的理想位置。

举个例子：

```python
for optimizer in optimizers:
    optimizer.step()
    model.on_before_zero_grad(optimizer) # < ---- 调用
    optimizer.zero_grad()
```

#### on_fit_start()/on_fit_end()

在训练开始/结束时调用。如果在DDP上，则在每个进程上调用。

#### on_load_checkpoint()/on_save_checkpoint()

使模型有机会在state_dict存在之前/后加载某些内容。

#### on_pretrain_routine_start()/on_pretrain_routine_end()

- fit
- pretrain_routine start
- pretrain_routine end
- training_start

#### on_train_batch_start(batch, batch_idx, dataloader_idx)/on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

对于每个训练批次，在开始/结束时调用。

#### on_train_epoch_start()/on_train_epoch_end()

对于每轮训练，在开始/结束时调用。

#### on_validation_batch_start(batch, batch_idx, dataloader_idx)/on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

同上。

#### on_validation_epoch_start()/on_validation_epoch_end()

同上。

#### on_test_batch_start(batch, batch_idx, dataloader_idx)/on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

同上。

#### on_test_epoch_start()/on_test_epoch_end()

同上。

#### optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)

重写此方法可以调整Trainer调用每个优化器的默认方式 。默认情况下，每个优化程序都会一次调用Lightning step()，zero_grad()。参数如下：

- epoch (int) – 当前轮数。
- batch_idx (int) – 当前批次的索引。
- optimizer (Optimizer) –优化器。
- optimizer_idx (int) – 如果有多个优化器，则使用。
- optimizer_closure (Optional[Callable]) – 所有优化器的关闭。
- on_tpu (bool) – 是否为TPU。
- using_native_amp (bool) – 是否为自动混合精度。
- using_lbfgs (bool) – 匹配的优化器是否为lbfgs。

举个例子：

```python
# 学习率预热
def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    # 预热
    if self.trainer.global_step < 500:
        lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.learning_rate

    # 更新参数
    optimizer.step(closure=optimizer_closure)
    optimizer.zero_grad()
```

#### optimizer_zero_grad(epoch, batch_idx, optimizer, optimizer_idx)

#### prepare_data()

用于下载和准备数据。

#### setup(stage)

在训练和测试开始时调用。当您需要动态构建模型或对模型进行调整时，这是一个很好的选择。使用DDP时，每个进程都会调用此钩子。参数如下：

- stage (str) – ‘fit’或‘test’。

举个例子：

```python
class LitModel(...):
    def __init__(self):
        self.l1 = None

    def prepare_data(self):
        download_data()
        tokenize()

        self.something = else

    def setup(stage):
        data = Load_data(...)
        self.l1 = nn.Linear(28, data.num_classes)
```

#### tbptt_split_batch(batch, split_size)

使用经过时间的截断的反向传播时，必须沿时间维度拆分每个批次。默认情况下，Lightning会处理此问题，但对于自定义行为，请覆盖此功能。参数如下：

- batch (Tensor) – 当前批次。
- split_size (int) – 分割的大小。

举个例子：

```python
def tbptt_split_batch(self, batch, split_size):
  splits = []
  for t in range(0, time_dims[0], split_size):
      batch_split = []
      for i, x in enumerate(batch):
          if isinstance(x, torch.Tensor):
              split_x = x[:, t:t + split_size]
          elif isinstance(x, collections.Sequence):
              split_x = [None] * len(x)
              for batch_idx in range(len(x)):
                  split_x[batch_idx] = x[batch_idx][t:t + split_size]

          batch_split.append(split_x)

      splits.append(batch_split)

  return splits
```

#### teardown(stage)

在训练和测试结束时调用。参数如下：

- stage (str) – ‘fit’或‘test’。

#### train_dataloader()/val_dataloader()/test_dataloader()

- fit()
- …
- prepare_data()
- setup()
- train_dataloader()
- val_dataloader()
- test_dataloader()

#### transfer_batch_to_device(batch, device)

# 哪些需要掌握，哪些不需要

- 需要掌握的方法：
  - \_\_init\_\_/forword
  - training_step/training_step_end/training_epoch_end
  - validation_step/validation_step_end/validation_epoch_end
  - configure_optimizer
  - freeze/unfreeze
  - save_hyperparameters
- 需要掌握的属性：
  - current_epoch
  - device
  - hparams
  - precision
  - trainer

其他的需要时再看即可。