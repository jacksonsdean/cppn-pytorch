import os
import unittest
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.util import visualize_network

class TestCompression(unittest.TestCase):
    def test_compression(self):
        jpg_sizes = []
        png_sizes = []
        cppn_sizes = []
        for i in range(10):
            config = CPPNConfig()
            config.res_h, config.res_w = 256, 256
            config.node_agg = 'mean'
            cppn = ImageCPPN(config)
            for _ in range(10):
                cppn.mutate()
            
            before = cppn.get_image().cpu()
            ToPILImage()(before.permute(2, 0, 1)).save("_before.jpg")
            ToPILImage()(before.permute(2, 0, 1)).save("_before.png")
            jpg_sizes.append(os.path.getsize("_before.jpg"))
            png_sizes.append(os.path.getsize("_before.png"))

            # print("Before")
            # for cx in cppn.connection_genome.values():
                # print(cx.key, cx.weight, cx.enabled)
            # for nx in cppn.node_genome.values():
                # print(nx.key, nx.activation)
            visualize_network(cppn, save_name="_test_net_before.png")
            plt.show()
    
            cppn.compress("_test.cppn")
            cppn_sizes.append(os.path.getsize("_test.cppn"))
            del cppn
            del config
                        
            config = CPPNConfig()
            config.node_agg = 'mean'
            config.res_h, config.res_w = 256, 256
            loaded = ImageCPPN(config)
            loaded.decompress("_test.cppn")
           
            # print("After")
            # for cx in loaded.connection_genome.values():
            #     print(cx.key, cx.weight)
            # for nx in loaded.node_genome.values():
            #     print(nx.key, nx.activation)
            visualize_network(loaded, save_name="_test_net_after.png")
            plt.show()
           
            after = loaded.get_image().cpu()
            # ToPILImage()(after.permute(2, 0, 1)).save("_after.jpg")
           
            assert torch.isclose(before, after).all(), f"\nIteration {i} failed. Before: {before.shape}, After: {after.shape}. Difference: {((before - after)**2).mean()}"
        
        os.remove("_before.jpg")
        os.remove("_before.png")
        # os.remove("_after.jpg"
        os.remove("_test.cppn")
        
        print(f"\n\nJPG sizes: {sum(jpg_sizes)/len(jpg_sizes):.2f}")
        print(f"PNG sizes: {sum(png_sizes)/len(png_sizes):.2f}")
        print(f"CPPN sizes: {sum(cppn_sizes)/len(cppn_sizes):.2f}")
            
        
if __name__ == "__main__":
    unittest.main()
        
        
        
        
        







  