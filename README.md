### 环境
PyTorch和飞桨（PaddlePaddle）

### 数据位置
data目录下

### 运行方式
设置文件 config/default.yaml

    其中参数phase有train，eval，test，predict四种模式。
    训练设置为eval就可，测试设置为test，预测设置为predict。
    记得更改相应的文件位置

根据需求更改default.yaml文件后运行main文件即可

### 模型权重
work_dir/fsd_eval/runs-52-53404.pdparams

### 训练日志
work_dir/fsd_eval/log.txt
