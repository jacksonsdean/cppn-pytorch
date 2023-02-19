import copy
from typing import List

import torch
from functorch.compile import compiled_function, draw_graph, aot_function

from cppn_torch import CPPN
from cppn_torch.gene import NodeType
from cppn_torch.graph_util import feed_forward_layers, find_node_with_id, get_incoming_connections, hsv2rgb

class ImageCPPN(CPPN):
    def __init__(self, config=None, nodes=None, connections=None) -> None:
        super().__init__(config, nodes, connections)
   
    @property
    def image(self):
        return self.outputs
   
    def get_image(self, force_recalculate=False, override_h=None, override_w=None, extra_inputs=None):
        """Returns an image of the network.
            Extra inputs are (batch_size, num_extra_inputs)
        """
        assert self.config is not None, "Config is None."
        
        # apply size override
        if self.config.dirty:
            raise RuntimeError("config is dirty, did you forget to call .reconfig() after changing config?")
       
        if not extra_inputs is None:
            assert extra_inputs.shape[-1] == self.config.num_extra_inputs, f"Wrong number of extra inputs {extra_inputs.shape[-1]} != {self.config.num_extra_inputs}"
        
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
        if isinstance(self.outputs, torch.Tensor):
            assert self.outputs is not None
            recalculate = recalculate or extra_inputs is not None # TODO we could cache the extra inputs and check
            recalculate = recalculate or self.config.res_h == self.outputs.shape[0]
            recalculate = recalculate or self.config.res_w == self.outputs.shape[1]
        else:
            # no cached image
            recalculate = True

        if not recalculate:
            # return the cached image
            assert self.outputs is not None
            assert self.outputs.device == self.device, f"Image is on {self.outputs.device}, should be {self.device}"
            assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
            return self.outputs

        if self.config.allow_recurrent:
            # pixel by pixel (good for debugging/recurrent)
            self.outputs = self.get_image_data_serial(extra_inputs=extra_inputs)
        else:
            # whole image at once (100sx faster)
            self.outputs = self.forward(extra_inputs=extra_inputs)

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

    def forward_(self, extra_inputs=None):
        """Evaluate the network to get output data in parallel
            Extra inputs are (batch_size, num_extra_inputs)
        """
        assert self.config is not None, "Config is None."
        
        batch_size = 1 if extra_inputs is None else extra_inputs.shape[0]
        res_h, res_w = self.config.res_h, self.config.res_w
        
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
        
        assert type(self).constant_inputs is not None
        
        # evaluate CPPN
        self.outputs = super().forward_(extra_inputs=extra_inputs)


        if len(self.config.color_mode)>2:
            self.outputs  =  self.outputs.permute(1, 2, 3, 0) # move color axis to end
        else:
            self.outputs  = torch.reshape(self.outputs , (batch_size, res_h, res_w))

        if batch_size == 1:
            self.outputs  = self.outputs.squeeze(0) # remove batch dimension if batch size is 1
            # reshape the outputs to image shape
            if not len(self.config.color_mode)>2:
                self.outputs  = torch.reshape(self.outputs, (res_h, res_w))
        
        if self.config.normalize_outputs:
            self.normalize_image()
        
        assert str(self.outputs.device)== str(self.device), f"Image is on {self.outputs.device}, should be {self.device}"
        assert self.outputs.dtype == torch.float32, f"Image is {self.outputs.dtype}, should be float32"
        return self.outputs
    
    def normalize_image(self):
        """Normalize from outputs (any range) to 0 through 255 and convert to ints"""
        assert self.outputs is not None, "No image to normalize"
        assert self.outputs.dtype == torch.float32, f"Image is not float32, is {self.outputs.dtype}"
        assert str(self.outputs.device) == str(self.device), f"Image is on {self.outputs.device}, should be {self.device}"
        assert self.config is not None, "Config is None."
        
        max_value = torch.max(self.outputs)
        min_value = torch.min(self.outputs)
        image_range = max_value - min_value
        self.outputs = self.outputs - min_value
        self.outputs = self.outputs/(image_range+1e-8)

        if self.config.color_mode == 'HSL':
            # assume output is HSL and convert to RGB
            self.outputs = hsv2rgb(self.outputs) # convert to RGB
    
    # def __deepcopy__(self, memo):
    #     """Deep copy the network"""
    #     return super().__deepcopy__(memo)
        
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