import unittest
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from cppn_torch import CPPN, CPPNConfig

# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configured for XOR:
config = CPPNConfig()
config.device = torch.device(device)
config.res_h = 2
config.res_w = 2
config.num_inputs = 2
config.num_outputs = 1
config.coord_range = (0, 1)
config.color_mode = "X" # XOR

def create_population(size=10):
    return [CPPN(config) for _ in range(size)]

class TestEvolution(unittest.TestCase):
    
    def test_evolution(self):
        N = 50
        elitism = 1/10 # top tenth
        parents = create_population(N)
        Y = torch.Tensor([0,1,1,0]).view(-1,1).to(device).unsqueeze(0).repeat(N, 1, 1)
        # fitness_function = lambda x: 1.0-torch.mean((x.reshape(-1,1) - y)**2) # XOR

        history = []
        for _ in range(2000):
            X = torch.stack([p() for p in parents])
            X = torch.clamp(X, 0, 1)
            assert X.shape == (N, config.res_h, config.res_w)
            assert torch.isfinite(X).all()

            X = X.reshape(N, 4, 1)
            fits = 1.0-torch.mean((X - Y)**2, dim= (1,2)) # XOR
            for i, f in enumerate(fits):
                parents[i].fitness = f
            
            max_fit = torch.max(fits).cpu()
            sum_fit = torch.sum(fits)
            
            assert torch.isfinite(fits).all()
            assert torch.isfinite(sum_fit)

            # fitness proportional selection
            selection_probs = fits/sum_fit
            selection_probs = selection_probs / torch.sum(selection_probs)
            selection_probs = selection_probs.cpu().numpy()
            
            sorted_parents = sorted(parents, key=lambda x: x.fitness)
            children = sorted_parents[:int(elitism*len(parents))] # elitism
            
            while len(children) < N:
                p = np.random.choice(parents, p=selection_probs)
                c = p.clone(new_id=True)
                c.mutate()
                children.append(c)
        
            parents = children
            
            history.append(max_fit)
        
        plt.plot(history)
        plt.savefig("tests/_test.png")

    
if __name__ == "__main__":
    unittest.main()
