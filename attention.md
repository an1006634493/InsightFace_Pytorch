1. bcolz直接pip install，千万不要自己setup
2. mxnet_cu90版本问题，不要直接'pip install mxnet_cu90'，否则在'import mxnet'时会'Illegal instruction (core dumped)'.
应该'pip install mxnet-cu90==1.3.1b20181004'。
