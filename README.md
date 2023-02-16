# cppn_torch

## Installation
With CUDA: 

`pip install  git+https://github.com/jacksonsdean/cppn_torch.git --extra-index-url https://download.pytorch.org/whl/cu116`

## Usage

Basic
```python
from cppn_torch import CPPN
import matplotlib.pyplot as plt

net = CPPN()

image = net.get_image()
plt.imshow(image)
```

Change configuration
```python
from cppn_torch import CPPN, Config
import matplotlib.pyplot as plt

config = Config()
config.activations = ['sin']

net = CPPN(config)

image = net.get_image()
plt.imshow(image)
```