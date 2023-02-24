

import time
import unittest
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.util import visualize_network

class TestSGD(unittest.TestCase):
    def test_sgd(self):
        cppn = ImageCPPN()
        cppn.config.res_h, cppn.config.res_w = 128, 128
        cppn.config.with_grad = True
        cppn.reconfig()
        for _ in range(10):
            cppn.mutate()
            
        tar = torch.randn(128, 128, 3, device=cppn.device)
        cppn.prepare_optimizer()
        
        for _ in range(100):
            img = cppn.get_image()
            loss = ((img - tar)**2).mean()
            cppn.backward(loss)
            
        img = cppn.get_image()
        assert img.dtype == torch.float32, f"Image dtype is {img.dtype}, expected torch.float32"
        assert img.shape == (128, 128, 3), f"Image shape is {img.shape}, expected (128, 128, 3)"
        
        
if __name__ == "__main__":
    unittest.main()
        