

import copy
import time
import unittest
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from cppn_torch.activation_functions import *

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.util import visualize_network
from cppn_torch.graph_util import activate_population

class TestPopulation(unittest.TestCase):
    def test_population_activation(self):
        seed = 20
        res = 28
        num = 400
        
        config = CPPNConfig()
        config.normalize_outputs = True
        config.set_res(res)
        config.seed = seed
        const_inputs = ImageCPPN.initialize_inputs(
            config.res_h, 
            config.res_w,
            config.use_radial_distance,
            config.use_input_bias,
            2 + config.use_radial_distance + config.use_input_bias,
            device=config.device,
            coord_range=(-0.5, 0.5)
            )
        
        # original
        config = CPPNConfig()
        config.normalize_outputs = True
        config.set_res(res)
        config.seed = seed
        
        naive_population = [ImageCPPN(config) for _ in range(num)]
        naive_method = []
        for cppn in naive_population:
            cppn.mutate()
            cppn.mutate()
            cppn.mutate()

        pop_population = copy.deepcopy(naive_population)
        
        for i, (naive, population) in enumerate(zip(naive_population, pop_population)):
            for cxn, cxp in zip(sorted(naive.connection_genome.items()), sorted(population.connection_genome.items())):
                assert cxn[1].weight ==  cxp[1].weight
                assert cxn[1].enabled == cxp[1].enabled
                assert cxn[0] == cxp[0], f"{i} {cxn[0]} != {cxp[0]}"
                
        t = time.time()
        for cppn in naive_population:
            naive_method.append(cppn.get_image(inputs=const_inputs))
        print(f"Naive: {time.time() - t}")
            
        t = time.time()
        population_method = activate_population(pop_population, config, inputs=const_inputs)
        print(f"Population: {time.time() - t}")
        
        assert len(population_method) == num
        
        for i, (naive, population) in enumerate(zip(naive_method, population_method)):
            assert naive.shape == (res, res, 3)
            assert population.shape == (res, res, 3)
            if not torch.isclose(naive, population, rtol=1e-5, atol=1e-5).all():
                plt.imshow(naive.cpu().numpy())
                plt.savefig(f"naive_{i}.png")
                plt.imshow(population.cpu().numpy())
                plt.savefig(f"population_{i}.png")
                assert torch.isclose(naive, population, rtol=1e-5, atol=1e-5).all(), f"{i} naive != population, max: {(naive - population).abs().max()}, rmse: {(torch.sqrt((naive - population)**2)).mean()}"
        
        
    
if __name__ == "__main__":
    unittest.main()
        