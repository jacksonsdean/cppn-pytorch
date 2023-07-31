import copy
from typing import List

import torch
from torch import nn
from functorch.compile import compiled_function, draw_graph, aot_function
from typing import Optional, TypeVar, Union

from cppn_torch.config import CPPNConfig as Config
from cppn_torch import CPPN
from cppn_torch.gene import NodeType
from cppn_torch.graph_util import feed_forward_layers, find_node_with_id, get_incoming_connections, hsl2rgb_torch

from cppn_torch.normalization import *

class ImageCPPN(CPPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.imagenet_norm = None
        if self.config.normalize_outputs and "imagenet" in self.config.normalize_outputs:
            self.imagenet_norm = Normalization(self.device)
   
    @property
    def image(self):
        return self.outputs
   
    def get_image(self, inputs=None, force_recalculate=False, override_h=None, override_w=None, extra_inputs=None, no_aot=True, channel_first=True, override_activation_mode=None):
        """Returns an image of the network.
            Extra inputs are (batch_size, num_extra_inputs)
        """
        assert self.config is not None, "Config is None."

        if inputs is None and self.config.dirty:
            raise RuntimeError("config is dirty, did you forget to call .reconfig() after changing config?")
       
        if not extra_inputs is None:
            assert extra_inputs.shape[-1] == self.config.num_extra_inputs, f"Wrong number of extra inputs {extra_inputs.shape[-1]} != {self.config.num_extra_inputs}"
        
        # apply size override
        if override_h is not None:
            self.config.res_h = override_h
        if override_w is not None:
            self.config.res_w = override_w
        
        if self.config.dry_run:
            self.outputs = torch.rand((self.config.res_h, self.config.res_w, len(self.config.color_mode)), device=self.device)
            return self.outputs

        # decide if we need to recalculate the image
        recalculate = False
        recalculate = recalculate or force_recalculate
        if hasattr(self, "outputs") and isinstance(self.outputs, torch.Tensor):
            assert self.outputs is not None
            if len(self.config.color_mode) == 3:
                if (channel_first and not torch.argmin(torch.tensor(self.outputs.shape)).item() == 0):
                    self.outputs = self.outputs.permute(2, 0, 1)
                elif (not channel_first and not torch.argmin(torch.tensor(self.outputs.shape)).item() == len(self.outputs.shape) - 1):
                    self.outputs = self.outputs.permute(1, 2, 0)
                
            O = self.outputs
            recalculate = recalculate or extra_inputs is not None # TODO we could cache the extra inputs and check
            recalculate = recalculate or self.config.res_h == O.shape[0]
            recalculate = recalculate or self.config.res_w == O.shape[1]
        else:
            # no cached image
            recalculate = True

        if not recalculate:
            # return the cached image
            assert self.outputs is not None
            assert str(self.outputs.device) == str(self.device), f"Output is on {self.outputs.device}, should be {self.device}"
            
            assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
            
            return self.outputs

        if self.config.allow_recurrent:
            # pixel by pixel (good for debugging/recurrent)
            self.outputs = self.get_image_data_serial(extra_inputs=extra_inputs)
        else:
            # whole image at once (100sx faster)
            self.outputs = self.forward(inputs=inputs, extra_inputs=extra_inputs, no_aot=no_aot, channel_first=channel_first, override_activation_mode=override_activation_mode)
       
        assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
        assert str(self.outputs.device) == str(self.device), f"Image is on {self.outputs.device}, should be {self.device}"
        return self.outputs

    def get_image_data_serial(self, extra_inputs=None):
        """Evaluate the network to get image data by processing each pixel
        serially. Much slower than the parallel method, but required if the
        network has recurrent connections."""
        raise NotImplementedError("get_image_data_serial is not implemented")
        assert self.config is not None, "Config is None."
        res_h, res_w = self.config.res_h, self.config.res_w
        pixels: List[torch.Tensor] = []
        
        for x in torch.linspace(-.5, .5, res_w, dtype=torch.float32, device=self.device):
            for y in torch.linspace(-.5, .5, res_h,dtype=torch.float32, device=self.device):
                # slow
                outputs = self.eval([x, y])
                pixels.extend(outputs)
                
        pixels_tensor = torch.stack(pixels)
        if len(self.config.color_mode)>2:
            pixels_tensor = torch.reshape(pixels_tensor, (res_w, res_h, self.n_outputs))
        else:
            pixels_tensor = torch.reshape(pixels_tensor, (res_w, res_h))

        self.outputs = pixels_tensor
        return self.outputs

    def forward_(self, inputs=None, extra_inputs=None, channel_first=True, override_activation_mode=None):
        """Evaluate the network to get output data in parallel
            Extra inputs are (batch_size, num_extra_inputs)
        """
        assert self.config is not None, "Config is None."
        
        # if self.device!=inputs.device:
        #     self.to(inputs.device) # breaks computation graph
        
        
        batch_size = 1 if extra_inputs is None else extra_inputs.shape[0]
        res_h, res_w = self.config.res_h, self.config.res_w
        
        if inputs is None:
            # lazy initialize the constant inputs
            if type(self).constant_inputs is None or\
                type(self).constant_inputs.shape[0] != res_h or\
                type(self).constant_inputs.shape[1] != res_w or\
                type(self).constant_inputs.device != self.device:
                # initialize inputs if the resolution or device changed
                type(self).initialize_inputs(res_h, res_w,
                    self.config.use_radial_distance,
                    self.config.use_input_bias,
                    self.config.num_inputs,
                    self.device,
                    type=type(self))
                
            inputs = type(self).constant_inputs
            
        assert inputs is not None
        assert inputs.shape[2] == self.config.num_inputs
        assert str(inputs.device) == str(self.device), f"Inputs are on {inputs.device}, should be {self.device}"
        
            
        # evaluate CPPN
        self.outputs = super().forward_(inputs=inputs,
                                        extra_inputs=extra_inputs,
                                        channel_first=channel_first,
                                        override_activation_mode=override_activation_mode)

        if batch_size == 1:
            self.outputs  = self.outputs.squeeze(0) # remove batch dimension if batch size is 1
            # reshape the outputs to image shape
            if not len(self.config.color_mode)>2:
                self.outputs  = torch.reshape(self.outputs, (res_h, res_w))
        
        if self.config.normalize_outputs:
            self.normalize_image()

        self.clamp_image()
        
        
        if len(self.config.color_mode) > 2 and not channel_first:
            if batch_size == 1:
                self.outputs  =  self.outputs.permute(1, 2, 0) # move color axis to end
            else:
                self.outputs  =  self.outputs.permute(0, 2, 3, 1) # move color axis to end
        else:
            if batch_size > 1:
                self.outputs  = torch.reshape(self.outputs , (batch_size, res_h, res_w))
                
        assert str(self.outputs.device)== str(self.device), f"Image is on {self.outputs.device}, should be {self.device}"
        assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
        return self.outputs
    
    def clamp_image(self):
        assert self.outputs is not None, "No image to clamp"
        assert self.outputs.dtype == torch.float32, f"Image is not float32, is {self.outputs.dtype}"
        self.outputs = torch.clamp(self.outputs, 0, 1)
            
    def normalize_image(self):
        """Normalize from outputs (any range) to 0 through 1"""
        assert self.outputs is not None, "No image to normalize"
        assert self.outputs.dtype == torch.float32, f"Image is not float32, is {self.outputs.dtype}"
        assert str(self.outputs.device) == str(self.device), f"Image is on {self.outputs.device}, should be {self.device}"
        assert self.config is not None, "Config is None."
        assert self.config.normalize_outputs, "normalize_outputs is False or None"
        
        if self.config.normalize_outputs:
            self.outputs = handle_normalization(self.outputs, self.config.normalize_outputs, self.imagenet_norm)
        if self.config.color_mode == 'HSL':
            # assume output is HSL and convert to RGB
            self.outputs = hsl2rgb_torch(self.outputs)
        
    def __call__(self, inputs=None, force_recalculate=False, override_h=None, override_w=None, extra_inputs=None, no_aot=True, channel_first=True):
        return self.get_image(
                            inputs=inputs,
                            force_recalculate=force_recalculate,
                            override_h=override_h,
                            override_w=override_w,
                            extra_inputs=extra_inputs,
                            no_aot=no_aot,
                            channel_first=channel_first
                            )
    
    
if __name__ == '__main__':
    # run a test
    import matplotlib.pyplot as plt

    net = ImageCPPN()
    i = net.get_image().cpu()
    
    plt.imshow(i)
    plt.show()
    for _ in range(10):
        net.mutate()
        
    i = net.get_image().cpu()
    plt.imshow(i)
    plt.show()