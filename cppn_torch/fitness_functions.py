from piq.feature_extractors import InceptionV3
from piq import ssim as piq_ssim, SSIMLoss, GS, FID, psnr as piq_psnr, fsim as piq_fsim
from piq import ms_ssim as piq_ms_ssim
from piq.perceptual import DISTS as piq_dists
from piq.perceptual import LPIPS as piq_lpips
from piq import ContentLoss
import piq

from skimage.transform import resize
import numpy as np

import torch
from torchvision.transforms import Resize

def control(candidates, target):
   return torch.rand(len(candidates), dtype=torch.float32, device=target.device)

def correct_dims(candidates, target):
   f,r = candidates, target
   if len(f.shape) == 2:
      # unbatched L
      f = f.repeat(3, 1, 1) # to RGB
      f = f.unsqueeze(0) # create batch
   elif len(f.shape) == 3:
      # either batched L or unbatched RGB
      if min(f.shape) == 3:
         # color images
         if torch.argmin(torch.tensor(f.shape)) != 0:
            f = f.permute(2,0,1)
         f = f.unsqueeze(0) # batch
      else:
         # batched L
         f = torch.stack([x.repeat(3,1,1) for x in f])
   else:
      # color
      if f.shape[1] != 3:
         f = f.permute(0,3,1,2)
   if len(r.shape) == 2:
      # unbatched L
      r = r.repeat(3,1,1) # to RGB
      r = r.unsqueeze(0) # create batch
   elif len(r.shape) == 3:
      # either batched L or unbatched RGB
      if min(r.shape) == 3:
         # color images
         if torch.argmin(torch.tensor(r.shape)) != 0:
            # move color to front
            r = r.permute(2,0,1)
         r = r.unsqueeze(0) # batch
      else:
         # batched L
         r = torch.stack([x.repeat(3,1,1) for x in r])         
   else:
      # color
      if r.shape[1] != 3:
         r = r.permute(0,3,1,2)


   f = f.to(torch.float32)
   r = r.to(torch.float32)
   
   # pad to 32x32 if necessary
   if f.shape[2] < 32 or f.shape[3] < 32:
      f = Resize((32,32),antialias=True)(f)
   if r.shape[2] < 32 or r.shape[3] < 32:
      r = Resize((32,32),antialias=True)(r)

   if f.shape[0] !=1 and r.shape[0] == 1:
      # only one target in batch, repeat for comparison
      r = torch.stack([r.squeeze() for _ in range(f.shape[0])])

   return f,r

# def correct_dims(candidates, target):
 # faster but doesn't work
#    # concatenate tensors along batch dimension
#    tensor = torch.cat([candidates, target.unsqueeze(0)], dim=0)

#    # unbatch L tensors and convert to RGB
#    if len(tensor.shape) == 2:
#       tensor = tensor.repeat(3, 1, 1)

#    # unbatch or permute color tensors
#    elif len(tensor.shape) == 3:
#       # unbatched color tensor
#       if min(tensor.shape) == 3:
#             if torch.argmin(torch.tensor(tensor.shape)) != 0:
#                # move color to front
#                tensor = tensor.permute(2, 0, 1)
#       else:
#             # batched L tensor, convert to RGB
#             tensor = torch.stack([x.repeat(3, 1, 1) for x in tensor])

#    # convert to float32 and pad to 32x32 if necessary
#    tensor = tensor.to(torch.float32)
#    if tensor.shape[2] < 32 or tensor.shape[3] < 32:
#       tensor = Resize(32)(tensor)
      
#    # split concatenated tensor back into original tensors
#    candidates, target = tensor.split(candidates.size(0), dim=0)

#    # ensure both tensors have the same batch size
#    if candidates.shape[0] != target.shape[0]:
#       target = target.repeat(candidates.shape[0], 1, 1, 1)

#    return candidates, target


def assert_images(*images):
   for img in images:
      max_val = torch.max(img)
      # assert (max_val > 1 or max_val == 0), "Fitness function expects values in range [0,255]"
      # assert img.dtype == torch.uint8, "Fitness function expects uint8 images"
      assert img.dtype == torch.float32, "Fitness function expects float32 images"

# NOTE (FOR NEAT ONLY):
# for now these must all return positive values due to the way 
# NEAT calculates number offspring per species 

def empty(candidates, target):
   raise NotImplementedError("Fitness function not implemented")

# Why not use MSE: https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf
def mse(candidates, target, keep_grad=False):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   if keep_grad:
      loss = torch.mean((candidates - target)**2, dim=(1,2,3))
   else:
      with torch.no_grad():
         loss = torch.mean((candidates - target)**2, dim=(1,2,3))

   value = torch.tensor([1.0]*len(candidates), requires_grad=keep_grad, device=loss.device, dtype=loss.dtype)
   value = value - loss
   return value

def MSE_LOSS(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   return torch.mean((candidates - target)**2, dim=(1,2,3))


def test(candidates, target):
   return (candidates/255).mean() # should get all white

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


def dists(candidates, target):
   if "DISTS_INSTANCE" in globals().keys():
      dists_instance = globals()["DISTS_INSTANCE"]
   else:
      dists_instance = piq_dists(reduction='none')
      globals()["DISTS_INSTANCE"] = dists_instance
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   loss = dists_instance(candidates, target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   return value
   
def lpips(candidates, target):
   if "LPIPS_INSTANCE" in globals().keys():
      lpips_instance = globals()["LPIPS_INSTANCE"]
   else:
      lpips_instance = piq_lpips(reduction='none')
      globals()["LPIPS_INSTANCE"] = lpips_instance
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   loss = lpips_instance(candidates, target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   # clamp to > 0
   value = torch.max(value, torch.tensor(0.0).to(value))
   return value


def haarpsi(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq.haarpsi(candidates, target, data_range=1., reduction='none')
   return value

def dss(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq.dss(candidates, target, data_range=1., reduction='none')
   return value
   
def gmsd(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   loss = piq.gmsd(candidates, target, data_range=1., reduction='none')
   value = torch.tensor([.35]*len(candidates)).to(loss) - loss # usually 0.35 is the max
   return value

def mdsi(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq.mdsi(candidates, target, data_range=1., reduction='none')
   return value
def mdsiinverted(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = torch.tensor(1.0)-piq.mdsi(candidates, target, data_range=1., reduction='none')
   return value

def msssim(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   candidates = Resize((161,161))(candidates)
   target = Resize((161,161))(target)
   value = piq.multi_scale_ssim(candidates, target, data_range=1., reduction='none')
   return value

def style(candidates, target):
   # Computes distance between Gram matrices of feature maps
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   loss = piq.StyleLoss(feature_extractor="vgg16", layers=("relu3_3",), reduction="none")(candidates, target)
   value = -loss
   return value

def content(candidates, target):
   if "CONTENT_INSTANCE" in globals().keys():
      content_instance = globals()["CONTENT_INSTANCE"]
   else:
      content_instance = piq.ContentLoss(
        feature_extractor="vgg16", layers=("relu3_3",), reduction='none')
      globals()["CONTENT_INSTANCE"] = content_instance
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   loss = piq.ContentLoss( feature_extractor="vgg16", layers=("relu3_3",), reduction='none')(candidates,target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   return value

def pieAPP(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   candidates,target = Resize((128,128),antialias=True)(candidates), Resize((128,128),antialias=True)(target)
   loss = piq.PieAPP(reduction='none', stride=32)(candidates, target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   return value


"""The principle philosophy underlying the original SSIM
approach is that the human visual system is highly adapted to
extract structural information from visual scenes. (https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf pg. 105)"""
def ssim(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq_ssim(candidates, target, data_range=1.0, reduction='none')
   return value

def psnr(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq_psnr(candidates, target, data_range=1.0, reduction='none')
   value = value / 50.0 # max is normally 50 DB
   return value

def vif(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   candidates, target = Resize((41,41),antialias=True)(candidates), Resize((41,41),antialias=True)(target)
   value = piq.vif_p(candidates, target, data_range=1.0, reduction='none')
   return value

def vsi(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq.vsi(candidates, target, data_range=1.0, reduction='none')
   return value

def srsim(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   candidates, target = Resize((161,161),antialias=True)(candidates), Resize((161,161),antialias=True)(target)
   value = piq.srsim(candidates, target, data_range=1.0, reduction='none')
   return value

def fsim(candidates, target):
   assert_images(candidates, target)
   # candidates, target = correct_dims(candidates, target)
   value = piq_fsim(candidates, target, data_range=1.0, reduction='none')
   return value

def dhash_images(imgs, hash_size: int, h=True):
   """ 
   Calculate the dhash signature of a given image 
   Args:
      image: the image (np array) to calculate the signature for
      hash_size: hash size to use, signatures will be of length hash_size^2
   
   Returns:
      Image signature as Numpy n-dimensional array
   """
   if len(imgs[0].shape) > 2:
      # color image, convert to greyscale TODO
      R, G, B = imgs[:,:,:,0], imgs[:,:,:,1], imgs[:,:,:,2]
      imgs = 0.2989 * R + 0.5870 * G + 0.1140 * B
   resized = torch.zeros((imgs.shape[0], hash_size + 1, hash_size))
   for n,i in enumerate(imgs):
      resized[n,:,:] = torch.tensor(resize(imgs[n,:,:], (hash_size + 1, hash_size), anti_aliasing=True))

   # compute differences between columns
   if h:
      diff = resized[:,:, 1:] > resized[:,:, :-1]
   else:
      # vertical 
      diff = resized[:,1:, :] > resized[:,-1:, :]
   hashed = diff
   return hashed


            
hash_size = 50 # hash size (10)
def dhash(candidates, target):
      """ 
      Calculate the dhash signature of a given image 
      Args:
         image: the image (np array) to calculate the signature for
         hash_size: hash size to use, signatures will be of length hash_size^2
      
      Returns:
         Image signature as Numpy n-dimensional array
      """
      raise NotImplementedError()
      assert_images(candidates, target)
      # f,r = correct_dims(candidates, target)
      # candidates, target = f/255.0, r/255.0
      hashes0h = dhash_images(r, hash_size, True)
      hashes1h = dhash_images(f, hash_size, True)
      hashes0v = dhash_images(r, hash_size, False)
      hashes1v = dhash_images(f, hash_size, False)
      # TODO allow this to run on batches
      diff_h = torch.sum(hashes0h != hashes1h) / hashes0h[0].numel()
      diff_v = torch.sum(hashes0v != hashes1v) / hashes0v[0].numel()
      return (1.0 - (diff_v+diff_h) / 2).item()


def feature_set(candidate_images, train_image):
   raise NotImplementedError() # needs to be adjusted for torch/multiple images
   """To compare two images and calculate fitness, each is de-
   fined by a feature set that includes the grayscale value at each pixel
   location (at N1 × N2 pixels) and the gradient between adjacent
   pixel values. The candidate feature set is then scaled to correspond
   with the normalized target feature set (Woolley and Stanley, 2011)."""
   # c = candidate, t = target
   # d(c,t) = 1 −e^(−α|c−t|)
   # α=5 (modulation parameter)
   # error = average(d(c,t))
   # fitness = 1 − err(C, T)^2
   
   h = train_image.shape[0]
   w = train_image.shape[1]

   # move color to the end(expected by NP)
   train_image = train_image.permute(1,2,0)
   candidate_images = candidate_images.permute(1,2,0)

   train_images = train_image.unsqueeze(0) # add "batch" dim
   candidate_images = candidate_images.unsqueeze(0) # TODO
   batch_size = candidate_images.shape[0]

   # convert to numpy (TODO NO)
   train_images = train_images.cpu().numpy() / 255.0
   candidate_images = candidate_images.cpu().numpy() / 255.0

   alpha = 5
   if(len(candidate_images[0].shape) < 3):
      # greyscale
      [gradient0x, gradient0y] = np.gradient(train_images, axis=(1,2)) # Y axis, X axis
      [gradient1x, gradient1y] = np.gradient(candidate_images, axis=(1,2))
      
      c = np.array([train_images.reshape(1, h*w), gradient0x.reshape(1, h*w ), gradient0y.reshape(1, h*w)])
      t = np.array([candidate_images.reshape(batch_size, h*w), gradient1x.reshape(batch_size, h*w), gradient1y.reshape(batch_size, h*w)])
      
      c = np.transpose(c, (1, 0, 2))# batch dimension first
      t = np.transpose(t, (1, 0, 2))
      diffs = c-t
      diffs = diffs.reshape(batch_size, 3*h*w) # flatten diffs (3 diffs, height, width)
      D = 1 - np.exp(-alpha * np.abs(diffs))
      diffs = np.mean(D, axis =1)

   else:
      [Y0, X0, C0] = np.gradient(train_images, axis=(1, 2, 3)) # gradient over all axes (Y axis, X axis, color axis)
      [Y1, X1, C1] = np.gradient(candidate_images, axis=(1, 2, 3))
      flat_dim = h*w*3 # flatten along all dimensions besides batch dim
      # math_module = np if using_cupy else np # 
      math_module = np
      c = math_module.array([train_images.reshape(1, flat_dim), Y0.reshape(1, flat_dim ), X0.reshape(1, flat_dim), C0.reshape(1, flat_dim)])
      t = math_module.array([candidate_images.reshape(batch_size, flat_dim), Y1.reshape(batch_size, flat_dim), X1.reshape(batch_size, flat_dim), C1.reshape(batch_size, flat_dim)])
      c = math_module.transpose(c, (1, 0, 2)) # batch dimension first
      t = math_module.transpose(t, (1, 0, 2)) 
      
      diffs = math_module.abs(t-c)
      
      diffs = diffs.reshape(batch_size, 4*3*h*w) # flatten diffs (4 diffs, 3 color channels, height, width)
      D = 1 - math_module.exp(-alpha * math_module.abs(diffs))
      diffs = math_module.mean(D, axis =1)
      
      

   return diffs



def average(candidates, target):
   candidates, target = candidates, target
   # fns = [ssim]
   fns =   [psnr,
            mse,
            lpips,
            dists,
            ssim,
            # style, # out of range
            fsim,
            mdsi,
            haarpsi,
            dss,
            vsi,
            multi_scale_ssim,
            gmsd,
            # content # needs testing
            # pieAPP # needs testing
            vif # needs testing
            # srsim # needs testing
            # dhash # needs testing
            # feature_set # needs testing
            # average # needs testing
            ]
   return sum([fn(candidates,target) for fn in fns]) / len(fns)


