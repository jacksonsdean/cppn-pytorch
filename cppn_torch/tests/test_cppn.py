import time
import unittest
import torch

from cppn_torch import CPPN, CPPNConfig

class TestCPPN(unittest.TestCase):
    def test_speed(self):
        SEED = 0
        s = time.time()
        config = CPPNConfig()
        config.seed = SEED
        config.res_h, config.res_w = 1024, 1024
        cppn = CPPN(config)
        for _ in range(30):
            cppn.mutate()
            cppn.add_node()
            cppn.add_connection()
            cppn.add_node()
            
        n = 300
        for _ in range(n):
            _ = cppn.forward() 
            
        
        print(f"Generated {n} 1024x1024 outputs in: {time.time() - s}")
    
    def test_image(self):
        config = CPPNConfig()
        config.set_res(128)
        cppn = CPPN(config)
        image = cppn.forward() 
        
        assert image.shape == (128, 128, 3), f"Image shape is {image.shape}, expected (128, 128, 3)"
        assert image.dtype == torch.float32, f"Image dtype is {image.dtype}, expected torch.float32"           
    
    def test_aggs(self):
        aggs = ["sum", "mean", "max", "min"]
        for agg in aggs:
            config = CPPNConfig()
            config.node_agg = agg
            cppn = CPPN(config)
            output = cppn.forward()
            assert output.shape == (config.res_h, config.res_w, 3), f"Output shape is {output.shape}, expected {(config.res_h, config.res_w, 3)}"
        
    def test_seed(self):
        SEED = 0

        config = CPPNConfig()
        config.seed = SEED
        cppn = CPPN(config)
        image_0 = cppn.forward() 
        
        del cppn
        
        config = CPPNConfig()
        config.seed = SEED
        cppn = CPPN(config)
        image_1 = cppn.forward()
        
        assert torch.isclose(image_0, image_1).all(), f"Images are not close. Difference: {((image_0 - image_1)**2).mean()}"
        
        
        
if __name__ == "__main__":
    unittest.main()
        