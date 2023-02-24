import time
import unittest
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.util import visualize_network

class TestImageCPPN(unittest.TestCase):
    def test_speed(self):
        SEED = 0
        s = time.time()
        config = CPPNConfig()
        config.seed = SEED
        config.res_h, config.res_w = 1024, 1024
        cppn = ImageCPPN(config)
        for _ in range(30):
            cppn.mutate()
            cppn.add_node()
            cppn.add_connection()
            cppn.add_node()
            
        n = 300
        for _ in range(n):
            _ = cppn.get_image(force_recalculate=True) 
            
        
        print(f"Generated {n} 1024x1024 images in: {time.time() - s}")
    
    def test_image(self):
        config = CPPNConfig()
        config.set_res(128)
        cppn = ImageCPPN(config)
        image = cppn.get_image() 
        
        assert image.shape == (128, 128, 3), f"Image shape is {image.shape}, expected (128, 128, 3)"
        assert image.dtype == torch.float32, f"Image dtype is {image.dtype}, expected torch.float32"           
    
        
    def test_seed(self):
        SEED = 0

        config = CPPNConfig()
        config.seed = SEED
        cppn = ImageCPPN(config)
        image_0 = cppn.get_image() 
        
        del cppn
        
        config = CPPNConfig()
        config.seed = SEED
        cppn = ImageCPPN(config)
        image_1 = cppn.get_image()
        
        assert torch.isclose(image_0, image_1).all(), f"Images are not close. Difference: {((image_0 - image_1)**2).mean()}"
        
        
        
if __name__ == "__main__":
    unittest.main()
        