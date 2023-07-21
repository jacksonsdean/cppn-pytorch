import random
import time
import unittest
import torch
from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.activation_functions import *

class TestImageCPPN(unittest.TestCase):
    def test_speed(self):
        return
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
    
    def test_activation_mode(self):
        n = 100
        r = 1024
        SEED = 0
        
        
        config = CPPNConfig()
        config.activation_mode = 'node'
        config.seed = SEED
        config.set_res(r)
        cppn = ImageCPPN(config)
        img0 = cppn.get_image()
                
        config = CPPNConfig()
        config.activation_mode = 'layer'
        config.seed = SEED
        config.set_res(r)
        cppn = ImageCPPN(config)
        img1 = cppn.get_image()
        
        assert torch.allclose(img0, img1), "Images are not the same"
        
        
        
        config = CPPNConfig()
        config.activation_mode = 'layer'
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
        for _ in range(n):
            _ = cppn.get_image(force_recalculate=True) 
        
        print(f"\nLAYER Generated {n} {r}x{r} images in: {time.time() - s}")
        
        config = CPPNConfig()
        config.activation_mode = 'node'
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
        for _ in range(n):
            _ = cppn.get_image(force_recalculate=True) 
        
        print(f"NODE Generated {n} {r}x{r} images in: {time.time() - s}")
        
    
    def test_image(self):
        return
        config = CPPNConfig()
        config.set_res(128)
        cppn = ImageCPPN(config)
        image = cppn.get_image() 
        
        assert image.shape == (128, 128, 3), f"Image shape is {image.shape}, expected (128, 128, 3)"
        assert image.dtype == torch.float32, f"Image dtype is {image.dtype}, expected torch.float32"           
    
    def test_num_activations(self):
        return
        all_fns = get_all()
        SEED = 0
        config = CPPNConfig()
        config.seed = SEED
        config.activations = all_fns
        config.set_res(32)
        
        const_inputs = ImageCPPN.initialize_inputs(
            config.res_h, 
            config.res_w,
            config.use_radial_distance,
            config.use_input_bias,
            2 + config.use_radial_distance + config.use_input_bias,
            device=config.device,
            coord_range=(-0.5, 0.5)
        )
        config = CPPNConfig()
        config.seed = SEED
        config.activations = all_fns
        config.set_res(32)
        s = time.time()
        for _ in range(300):
            cppn = ImageCPPN(config)
            for _ in range(4):
                cppn.mutate()
            image = cppn.get_image(const_inputs)
        print(f"Generated 300 32x32 images with {len(config.activations)} activations in: {time.time() - s}")
        
        config = CPPNConfig()
        config.seed = SEED
        config.activations = random.sample(all_fns, 7)
        config.set_res(32)
        s = time.time()
        for _ in range(300):
            cppn = ImageCPPN(config)
            for _ in range(4):
                cppn.mutate()
            image = cppn.get_image(const_inputs)
        print(f"Generated 300 32x32 images with {len(config.activations)} activations in: {time.time() - s}")
    
        config = CPPNConfig()
        config.seed = SEED
        config.activations = random.sample(all_fns, 3)
        config.set_res(32)
        s = time.time()
        for _ in range(300):
            cppn = ImageCPPN(config)
            for _ in range(4):
                cppn.mutate()
            image = cppn.get_image(const_inputs)
        print(f"Generated 300 32x32 images with {len(config.activations)} activations in: {time.time() - s}")
       
        config = CPPNConfig()
        config.seed = SEED
        config.activations = random.sample(all_fns, 1)
        config.set_res(32)
        s = time.time()
        for _ in range(300):
            cppn = ImageCPPN(config)
            for _ in range(4):
                cppn.mutate()
            image = cppn.get_image(const_inputs)
        print(f"Generated 300 32x32 images with {len(config.activations)} activations in: {time.time() - s}")
    
    def test_aggs(self):
        return
        aggs = ["sum", "mean", "max", "min"]
        for agg in aggs:
            config = CPPNConfig()
            config.node_agg = agg
            cppn = ImageCPPN(config)
            output = cppn.get_image()
            assert output.shape == (config.res_h, config.res_w, 3), f"Output shape is {output.shape}, expected {(config.res_h, config.res_w, 3)}"
        
    def test_seed(self):
        return
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
        