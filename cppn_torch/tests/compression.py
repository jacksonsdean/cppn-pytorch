import os
import unittest
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

from cppn_torch import ImageCPPN, CPPNConfig

class TestCompression(unittest.TestCase):
    def test_compression(self):
        jpg_sizes = []
        png_sizes = []
        cppn_sizes = []
        for i in range(10):
            config = CPPNConfig()
            config.res_h, config.res_w = 256, 256
            cppn = ImageCPPN(config)
            for _ in range(10):
                cppn.mutate()
            
            before = cppn.get_image().cpu()
            ToPILImage()(before.permute(2, 0, 1)).save("_before.jpg")
            ToPILImage()(before.permute(2, 0, 1)).save("_before.png")
            jpg_sizes.append(os.path.getsize("_before.jpg"))
            png_sizes.append(os.path.getsize("_before.png"))
            cppn.compress("_test.cppn")
            cppn_sizes.append(os.path.getsize("_test.cppn"))
            del cppn
            del config
                        
            config = CPPNConfig()
            config.res_h, config.res_w = 256, 256
            loaded = ImageCPPN(config)
            loaded.decompress("_test.cppn")
            after = loaded.get_image().cpu()
            # ToPILImage()(after.permute(2, 0, 1)).save("_after.jpg")
            assert (before == after).all(), f"Iteration {i} failed. Before: {before.shape}, After: {after.shape}. Difference: {((before - after)**2).mean()}"
        
        os.remove("_before.jpg")
        os.remove("_before.png")
        os.remove("_test.cppn")
        
        print(f"JPG sizes: {sum(jpg_sizes)/len(jpg_sizes):.2f}")
        print(f"PNG sizes: {sum(png_sizes)/len(png_sizes):.2f}")
        print(f"CPPN sizes: {sum(cppn_sizes)/len(cppn_sizes):.2f}")
            
        
if __name__ == "__main__":
    unittest.main()
        
        
        
        
        







  