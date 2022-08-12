from piq.feature_extractors import InceptionV3
from piq import ssim, SSIMLoss, GS, FID, psnr
from piq import haarpsi as hwbpsi
import torch
from torchvision.transforms import Resize


# NOTE:
# for now these must all return positive values due to the way 
# NEAT calculates number offspring per species 

def mse(candidate, target):
   return (255.0**2-((target-candidate)**2).mean()) / 255.0**2

def test(candidate, target):
   return (candidate/255).mean() # should get all white

def xor(cppn):
   inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
   targets = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
   fitnesses=[]
   for i, _ in enumerate(inputs):
      cppn.reset_activations(False)
      outputs = cppn.eval(inputs[i])
      outputs = outputs[0] # one output node
      assert outputs.shape == targets[i].shape, "Output shape {} != Target shape {}".format(outputs.shape, targets[i].shape)
      fitnesses.append(abs(outputs-targets[i]))
      # print("input:", inputs[i], "output:", outputs, "target:", targets[i], "fit:", fitnesses[-1])
   assert len(fitnesses) == 4
   return max(torch.tensor(0,dtype=torch.float32), 100 - torch.sum(torch.stack(fitnesses)))

def haarpsi(candidate, target):
   hwbpsi_value = 0
   f,r = candidate, target
   f,r = f.unsqueeze(0), r.unsqueeze(0)
   
   f,r = f/255, r/255
   
   # pad to 32x32 if necessary
   if f.shape[2] < 32 or f.shape[3] < 32:
      f = Resize(32)(f)
   if r.shape[2] < 32 or r.shape[3] < 32:
      r = Resize(32)(r)
   
   if(r.size()[1]!=3):
      r = r.repeat(1, 3, 1, 1)
   if(f.size()[1]!=3):
      f = f.repeat(1, 3, 1, 1)
   # calculate:
   hwbpsi_value = hwbpsi(f, r)
   return hwbpsi_value
