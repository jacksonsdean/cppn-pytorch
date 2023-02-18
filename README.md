# cppn_torch

## Installation
With CUDA: 

`pip install  git+https://github.com/jacksonsdean/cppn_torch.git --extra-index-url https://download.pytorch.org/whl/cu116`

## Usage

Basic
```python
from cppn_torch import ImageCPPN
import matplotlib.pyplot as plt

net = ImageCPPN()

image = net.get_image()
plt.imshow(image.cpu())
```

Change configuration
```python
from cppn_torch import ImageCPPN, Config
import matplotlib.pyplot as plt

config = Config()
config.activations = ['sin']

net = ImageCPPN(config)

image = net.get_image()
plt.imshow(image.cpu())
```