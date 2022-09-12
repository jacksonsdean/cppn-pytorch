from piq.feature_extractors import InceptionV3
from piq import ssim as piq_ssim, SSIMLoss, GS, FID, psnr as piq_psnr, fsim as piq_fsim
from piq import haarpsi as hwbpsi
from piq import ms_ssim as piq_ms_ssim
from piq.perceptual import DISTS as piq_dists
from piq.perceptual import LPIPS as piq_lpips



import torch
from torchvision.transforms import Resize

def correct_dims(candidate, target):
   f,r = candidate, target
   
   # change to 3 channels if necessary
   if len(r.shape) == 2:
      r = r.unsqueeze(0)
      r = r.repeat(3,1,1)
   elif torch.argmin(torch.tensor(r.shape)) != 0:
      # color images, move channels to front
      r = r.permute(2,0,1)
      
   if len(f.shape) == 2:
      f = f.unsqueeze(0)
      f = f.repeat(3,1,1)
   elif torch.argmin(torch.tensor(f.shape)) != 0:
      # color images, move channels to front
      f = f.permute(2,0,1)
   
   # fake batch
   if len(f.shape) == 3:
      f = f.unsqueeze(0)
   if len(r.shape) == 3:
      r = r.unsqueeze(0) 

   if f.max() > 1:
      f = f/255.0
   if r.max() > 1:
      r = r/255.0
   f = f.to(torch.float32)
   r = r.to(torch.float32)
   
   # pad to 32x32 if necessary
   if f.shape[1] < 32 or f.shape[2] < 32:
      f = Resize((32,32))(f)
   if r.shape[1] < 32 or r.shape[2] < 32:
      r = Resize((32,32))(r)
      
   return f,r

def assert_images(*images):
   for img in images:
      max_val = torch.max(img)
      assert (max_val > 1 or max_val == 0), "Fitness function expects values in range [0,255]"
      assert img.dtype == torch.uint8, "Fitness function expects uint8 images"

# NOTE:
# for now these must all return positive values due to the way 
# NEAT calculates number offspring per species 

def empty(candidate, target):
   raise NotImplementedError("Fitness function not implemented")

# Why not use MSE: https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf
def mse(candidate, target):
   assert_images(candidate, target)
   candidate,target = candidate.to(torch.float32), target.to(torch.float32)
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




def dists(candidate, target):
   if "DISTS_INSTANCE" in globals().keys():
      dists_instance = globals()["DISTS_INSTANCE"]
   else:
      dists_instance = piq_dists()
      globals()["DISTS_INSTANCE"] = dists_instance
   assert_images(candidate, target)
   f, r = correct_dims(candidate, target)
   value = dists_instance(f, r)
   if value < 0:
      return torch.tensor(0.0)
   return value

def haarpsi(candidate, target):
   assert_images(candidate, target)
   f, r = correct_dims(candidate, target)
   # calculate:
   return hwbpsi(f, r)


"""The principle philosophy underlying the original SSIM
approach is that the human visual system is highly adapted to
extract structural information from visual scenes. (https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf pg. 105)"""
def ssim(candidate, target):
   assert_images(candidate, target)
   f, r = correct_dims(candidate, target)
   value = piq_ssim(f, r, data_range=1.0, reduction='mean')
   if value < 0:
      return torch.tensor(0.0)
   return value

def psnr(candidate, target):
   assert_images(candidate, target)
   f, r = correct_dims(candidate, target)
   value = piq_psnr(f, r, data_range=1.0, reduction='mean')
   value = value / 50.0 # max is normally 50 DB
   if value < 0:
      return torch.tensor(0.0)
   return value

def fsim(candidate, target):
   assert_images(candidate, target)
   f, r = correct_dims(candidate, target)
   value = piq_fsim(f, r)
   if value < 0:
      return torch.tensor(0.0)
   return value

def average(candidate, target):
   f, r = candidate, target
   # fns = [ssim]
   fns = [mse, ssim, psnr, fsim, haarpsi]
   return sum([fn(f,r) for fn in fns]) / len(fns)