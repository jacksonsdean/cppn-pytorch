

import time
import unittest
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from cppn_torch.activation_functions import *

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.util import visualize_network

class TestSGD(unittest.TestCase):
    def test_sgd(self):
        cppn = ImageCPPN()
        cppn.config.res_h, cppn.config.res_w = 128, 128
        cppn.config.with_grad = True
        cppn.config.node_agg = "sum"
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
        
        
    def test_activation_mode(self):
        r = 1024
        SEED = 0
        
        config = CPPNConfig()
        config.with_grad = True
        config.activation_mode = 'layer'
        config.activations = [sin]
        config.seed = SEED
        config.set_res(r)
        cppn = ImageCPPN(config)
        tar = torch.randn(r, r, 3, device=cppn.device)
        for _ in range(30):
            cppn.mutate()
            cppn.add_node()
            cppn.add_connection()
            cppn.add_node()
            
        s = time.time()
        cppn.prepare_optimizer()
        for _ in range(100):
            img = cppn.get_image()
            loss = ((img - tar)**2).mean()
            cppn.backward(loss)
        print(f"\nLAYER {time.time() - s}")
        
        
        config = CPPNConfig()
        config.activation_mode = 'node'
        config.with_grad = True
        config.activations = [sin]
        config.seed = SEED
        config.set_res(r)
        cppn = ImageCPPN(config)
        for _ in range(30):
            cppn.mutate()
            cppn.add_node()
            cppn.add_connection()
            cppn.add_node()

        s = time.time()
        cppn.prepare_optimizer()
        for _ in range(100):
            img = cppn.get_image()
            loss = ((img - tar)**2).mean()
            cppn.backward(loss)
        print(f"\nNODE {time.time() - s}")
        
    
        
    
if __name__ == "__main__":
    unittest.main()
        