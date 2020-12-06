[TOC]

# CSVLogger

抛弃Pytorch Lightning自带的logger，自定义logger。

## 修改LightningModule

```python
class CustomModel(pl.LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.model = ...
        # 用于计算loss
        self.train_criterion = CrossEntropyLoss()
        self.val_criterion = CrossEntropyLoss()
        # 用于计算metric
        self.train_metric = ClassificationMetric()
        self.val_metric = ClassificationMetric()
        # 用于保存log
        self.history = {
            "loss": [], "acc": [],
            "val_loss": [], "val_acc": [],
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        # 计算loss
        loss = self.train_criterion(_y, y)
        # 统计结果
        self.train_metric.update(_y, y)
        return loss

    def training_epoch_end(self, outs):
        # 计算平均loss
        loss = 0.
        for out in outs:
            loss += out["loss"].cpu().detach().item()
        loss /= len(outs)
        # 计算指标
        acc = self.train_metric.compute()
        # 保存log
        self.history["loss"].append(loss)
        self.history["acc"].append(acc)

    def validation_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        # 计算loss
        val_loss = self.val_criterion(_y, y)
        # 统计结果
        self.val_metric.update(_y, y)
        return val_loss

    def validation_epoch_end(self, outs):
        # 计算平均loss
        val_loss = sum(outs).item() / len(outs)
        # 计算指标
        val_acc1 = self.val_metric.compute()
        # 保存log
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        scheduler = ...
        return [optimizer], [scheduler]
```

## 自定义Callback

```python
class CSVLogger(Callback):
    def __init__(self, dirpath="history/", filename="history"):
        super(CSVLogger, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".csv":
            self.name += ".csv"

    def on_epoch_end(self, trainer, module): # 在每轮结束时保存log到磁盘
        history = pd.DataFrame(module.history)
        history.to_csv(self.name, index=False)
```

# ModelCheckpoint

模型检查点，尽管Pytorch Lightning官方有实现，我们依旧可以自定义一个。

## 修改LightningModule

和CSVLogger的一样，主要是history记录log。

## 自定义Callback

```python
class ModelCheckpoint(Callback):
    def __init__(self, dirpath="checkpoint/", filename="checkpoint", monitor="val_acc", mode="max"):
        super(ModelCheckpoint, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".pth":
            self.name += ".pth"
        self.monitor = monitor
        self.mode = mode
        self.value = 0. if mode == "max" else 1e6

    def on_epoch_end(self, trainer, module): # 在每轮结束时检查
        if self.mode == "max" and module.history[self.monitor][-1] > self.value:
            self.value = module.history[self.monitor][-1]
            torch.save(module.state_dict(), self.name)
        if self.mode == "min" and module.history[self.monitor][-1] < self.value:
            self.value = module.history[self.monitor][-1]
            torch.save(module.state_dict(), self.name)
```

# LearningCurve

我们来画个学习曲线，看看训练的各个指标的趋势。

## 修改LightningModule

和CSVLogger的一样，主要是history记录log。

## 自定义Callback

```python
class LearningCurve(Callback):
    def __init__(self, dirpath="checkpoint/", filename="log", figsize=(12, 4), names=("loss", "acc", "f1")):
        super(LearningCurve, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".png":
            self.name += ".png"
        self.figsize = figsize
        self.names = names

    def on_fit_end(self, trainer, module): # 在.fit结束时画图
        history = module.history
        plt.figure(figsize=self.figsize)
        for i, j in enumerate(self.names):
            plt.subplot(1, len(self.names), i + 1)
            plt.title(j + "/val_" + j)
            plt.plot(history[j], "--o", color='r', label=j)
            plt.plot(history["val_" + j], "-*", color='g', label="val_" + j)
            plt.legend()
        plt.savefig(self.name)
        plt.show()
```

# 注意事项

- 当你定义多个Callback时，一定要使他们不相关。
- 定义Callback时注意每个操作的调用时间顺序。
- 建议在LightningModule中定义一个同上的history用来保存log，而不是用官方的logger，这样可以避免很多bug，而且随时都能用上。