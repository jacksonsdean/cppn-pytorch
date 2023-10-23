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


imagenet_norm = None

class ImageCPPN(CPPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if 'imagenet' in self.normalize_outputs:
            global imagenet_norm
            if imagenet_norm is None:
                imagenet_norm = Normalization()
   
    @property
    def image(self):
        return self.outputs
   
    def get_image(self, inputs=None, force_recalculate=False, channel_first=True, act_mode='node'):
        """Returns an image of the network.
            Extra inputs are (batch_size, num_extra_inputs)
        """
        res_h, res_w = inputs.shape[1], inputs.shape[2]
        
        # decide if we need to recalculate the image
        recalculate = False
        recalculate = recalculate or force_recalculate
        if hasattr(self, "outputs") and isinstance(self.outputs, torch.Tensor):
            assert self.outputs is not None
            if len(self.color_mode) == 3:
                if (channel_first and not torch.argmin(torch.tensor(self.outputs.shape)).item() == 0):
                    self.outputs = self.outputs.permute(2, 0, 1)
                elif (not channel_first and not torch.argmin(torch.tensor(self.outputs.shape)).item() == len(self.outputs.shape) - 1):
                    self.outputs = self.outputs.permute(1, 2, 0)
                
            O = self.outputs
            recalculate = recalculate or O.shape[0] != res_h
            recalculate = recalculate or O.shape[1] != res_w
        else:
            # no cached image
            recalculate = True

        if not recalculate:
            # return the cached image
            assert self.outputs is not None
            assert str(self.outputs.device) == str(self.device), f"Output is on {self.outputs.device}, should be {self.device}"
            
            assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
            
            return self.outputs

        self.outputs = self.forward(inputs=inputs, channel_first=channel_first, act_mode=act_mode)
       
        assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
        assert str(self.outputs.device) == str(self.device), f"Image is on {self.outputs.device}, should be {self.device}"
        return self.outputs

    def get_image_data_serial(self, extra_inputs=None):
        """Evaluate the network to get image data by processing each pixel
        serially. Much slower than the parallel method, but required if the
        network has recurrent connections."""
        raise NotImplementedError("get_image_data_serial is not implemented")
        # TODO: would be necessary for recurrent networks

    def forward(self, inputs=None, channel_first=True, act_mode='node'):
        """Evaluate the network to get output data in parallel
            Extra inputs are (batch_size, num_extra_inputs)
        """
        assert inputs is not None
        assert inputs.shape[2] == self.n_in_nodes, f"Input shape is {inputs.shape}, should be (h, w, {self.n_in_nodes})"
        assert str(inputs.device) == str(self.device), f"Inputs are on {inputs.device}, should be {self.device}"
            
        # evaluate CPPN
        self.outputs = super().forward(inputs=inputs,
                                        channel_first=channel_first,
                                        act_mode=act_mode)

       
        if self.normalize_outputs:
            self.normalize_image()

        self.clamp_image()
                
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
        
        if self.normalize_outputs:
            self.outputs = handle_normalization(self.outputs, self.normalize_outputs, imagenet_norm)
        if self.color_mode == 'HSL':
            # assume output is HSL and convert to RGB
            self.outputs = hsl2rgb_torch(self.outputs)
        
    def __call__(self, inputs=None, force_recalculate=False, channel_first=True):
        return self.get_image(
                            inputs=inputs,
                            force_recalculate=force_recalculate,
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