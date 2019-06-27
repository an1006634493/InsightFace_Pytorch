1. When install `bcolz`, just directly use `pip install bcolz`, don not use `setup.py`.
2. When install `mxnet_cu90`, do not use `pip install mxnet_cu90`，otherwise when you `import mxnet`, there will be `Illegal instruction (core dumped)`. Please use `pip install mxnet-cu90==1.3.1b20181004`.
