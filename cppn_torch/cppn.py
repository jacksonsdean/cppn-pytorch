"""Contains the CPPN, Node, and Connection classes."""
import copy
from enum import IntEnum
from itertools import count
import math
import json
import os
import random
from typing import Callable, List
from typing import Union
from cppn_torch.graph_util import activate_layer
import torch
from torch.nn import ConvTranspose2d, Conv2d
import networkx as nx
import logging
from torch import nn
from functorch.compile import compiled_function, draw_graph, aot_function
from cppn_torch.activation_functions import identity
from cppn_torch.graph_util import *
from cppn_torch.config import CPPNConfig as Config
from cppn_torch.gene import * 
from cppn_torch.util import upscale_conv2d

def random_uniform(generator, low=0.0, high=1.0, grad=False):
    return ((low - high) * torch.rand(1, device=generator.device, requires_grad=grad, generator=generator) + high)[0]
def random_normal (generator, mean=0.0, std=1.0, grad=False):
    return torch.randn(1, device=generator.device, requires_grad=grad, generator=generator)[0] * std + mean

def random_choice(generator, choices, count, replace):
    if not replace:
        indxs = torch.randperm(len(choices))[:count]
        output = []
        for i in indxs:
            output.append(choices[i])
        return output
    else:
        return choices[torch.randint(len(choices), (count,), generator=generator)]


class Block(nn.Module):
    # TODO: batch norm bad idea for small batch sizes
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(Block, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            upscale_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(out_channels),
            )
        else:
          self.skip = None

        self.block = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            upscale_conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=1, bias=False),
            # nn.ReLU()
            
            )
        

    def forward(self, x):
        out = self.block(x)
        # out += (x if self.skip is None else self.skip(x))
        # out = torch.nn.functional.relu(out)
        return out

class CPPN():
    """A CPPN Object with Nodes and Connections."""

    constant_inputs = torch.zeros((0, 0, 0), dtype=torch.float32,requires_grad=False) # (res_h, res_w, n_inputs)
    current_id = 1 # 0 reserved for 'random' parent
    node_indexer = None
    
    
    @staticmethod
    def initialize_inputs(res_h, res_w, use_radial_dist, use_bias, n_inputs, device, coord_range=(-.5,.5), type=None, dtype=torch.float32):
        """Initializes the pixel inputs."""
        if type is None:
            type = __class__
        
        if not isinstance(coord_range[0], tuple):
            # assume it's a single range for both x and y
            coord_range_x = coord_range
            coord_range_y = coord_range
        else:
            coord_range_x, coord_range_y = coord_range
            
        # Pixel coordinates are linear within coord_range
        x_vals = torch.linspace(coord_range_x[0], coord_range_x[1], res_w, device=device,dtype=dtype)
        y_vals = torch.linspace(coord_range_y[0], coord_range_y[1], res_h, device=device,dtype=dtype)

        # initialize to 0s
        type.constant_inputs = torch.zeros((res_h, res_w, n_inputs), dtype=dtype, device=device, requires_grad=False)

        # assign values:
        type.constant_inputs[:, :, 0] = y_vals.unsqueeze(1).repeat(1, res_w)
        type.constant_inputs[:, :, 1] = x_vals.unsqueeze(0).repeat(res_h, 1)
            
        
        if use_radial_dist:
            # d = sqrt(x^2 + y^2)
            type.constant_inputs[:, :, 2] = torch.sqrt(type.constant_inputs[:, :, 0]**2 + type.constant_inputs[:, :, 1]**2)
        if use_bias:
            type.constant_inputs[:, :, -1] = torch.ones((res_h, res_w), dtype=dtype, device=device, requires_grad=False) # bias = 1.0
        
        repeat_dims = 2 # just y, x
        if use_radial_dist:
            repeat_dims += 1 # add radial dist
        n_repeats = 0   
        for i in range(n_repeats):
            type.constant_inputs  = torch.cat((type.constant_inputs, type.constant_inputs[:, :, :repeat_dims]), dim=2)
        
        return type.constant_inputs

    def __init__(self, config = Config(), nodes = None, connections = None) -> None:
        """Initialize a CPPN."""
        self.config = config
            
        assert self.config is not None
        self.outputs = None
        self.node_genome = {}
        self.connection_genome = {}
        self.selected = False
        self.species_id = 0
        self.id = type(self).get_id()
        
        self.in_dim = copy.deepcopy(self.config.num_inputs)
        self.reconfig(self.config, nodes, connections)
        
        self.parents = (0, 0)
        self.fitness = torch.tensor(-torch.inf, device=self.device)
        self.novelty = torch.tensor(-torch.inf, device=self.device)
        self.adjusted_fitness = torch.tensor(-torch.inf, device=self.device)
        
        self.age = 0
    
    def configure_generator(self):
        if self.config.seed is not None:
            self.seed = (self.config.seed + self.id) % 2**32
        else:
            self.seed = random.randint(0, 2**32 - 1)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

    
    def reconfig(self, config = None, nodes = None, connections = None):
        if config is not None:
            self.config = config
        assert self.config is not None
     
        self.device = self.config.device
            
        if self.device is None:
            raise ValueError("device is None") # TODO
            # no device specified, try to use GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.configure_generator()
        
        self.n_outputs = len(self.config.color_mode) # RGB: 3, HSV: 3, L: 1
        
        self.pre_layers = []
        
        self.n_in_nodes = self.config.num_inputs
        
        if self.config.num_conv > 0:
            last_c = 8
            self.pre_layers.append(Conv2d(self.config.num_inputs, last_c, kernel_size=5, stride=1, padding=2, bias=False).to(self.device))
            self.pre_layers[-1] = self.pre_layers[-1].to(self.device)
            for _ in range(self.config.num_conv-1):
                self.pre_layers.append(Conv2d(last_c, 3, kernel_size=5, stride=1, padding=2, bias=False))
                last_c = 3
                self.pre_layers[-1] = self.pre_layers[-1].to(self.device)
            self.n_in_nodes = last_c
            
        if nodes is None:
            self.initialize_node_genome()
        else:
            self.node_genome = nodes
        if connections is None:
            self.initialize_connection_genome()
        else:
            self.connection_genome = connections
        
        F = 3
        S = 1
        P = 1
        
        
        self.post_layers = []
        num_channels = len(self.config.color_mode)
        flat_size = self.config.res_h*self.config.res_w*num_channels
        
        for _ in range(self.config.num_upsamples):
            self.post_layers.append(Block(num_channels, num_channels))
            self.post_layers[-1] = self.post_layers[-1].to(self.device)
     
        for _ in range(self.config.num_post_conv):
            self.post_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True))
            self.post_layers[-1] = self.post_layers[-1].to(self.device)
        
        self.config._not_dirty()
    
    @staticmethod
    def get_id():
        __class__.current_id += 1
        return __class__.current_id - 1

    def get_new_node_id(self):
        """Returns a new node id`."""
        if type(self).node_indexer is None:
            if self.node_genome == {}:
                return 0
            if self.node_genome is not None:
                type(self).node_indexer = count(max(list(self.node_genome)) + 1)
            else:
                type(self).node_indexer = count(max(list(self.node_genome)) + 1)

        new_id = next(type(self).node_indexer)
        assert new_id not in self.node_genome.keys()
        return new_id
    
    def set_id(self, id):
        self.id = id
        
    def initialize_node_genome(self):
        """Initializes the node genome."""
        assert self.config is not None
        n_in = self.n_in_nodes
        n_out = self.n_outputs  

        total_node_count = n_in + n_out + self.config.hidden_nodes_at_start
        
        for idx in range(n_in):
            fn = identity if not self.config.allow_input_activation_mutation else choose_random_function(self.generator, self.config)
            new_node = Node(-(1+idx), fn, NodeType.INPUT, 0, self.config.node_agg, self.device, grad=self.config.with_grad)
            self.node_genome[new_node.key] = new_node
            
        for idx in range(n_in, n_in + n_out):
            if self.config.output_activation is None:
                output_fn = choose_random_function(self.generator, self.config)
            else:
                output_fn = self.config.output_activation
            new_node = Node(-(1+idx), output_fn, NodeType.OUTPUT, 2, self.config.node_agg, self.device, grad=self.config.with_grad)
            self.node_genome[new_node.key] = new_node
        
        # the rest are hidden:
        for _ in range(n_in + n_out, total_node_count):
            new_node = Node(self.get_new_node_id(), choose_random_function(self.generator, self.config),
                            NodeType.HIDDEN, 1, self.config.node_agg, self.device, grad=self.config.with_grad)
            self.node_genome[new_node.key] = new_node
     
     
    @property
    def params(self):
        return self.get_params()
    
    def get_cppn_params(self):
        params = []
        required_nodes = required_for_output(*get_ids_from_individual(self))
        
        # cxs that end at a required node
        required_cxs = set()
        for node_id in required_nodes:
            for cx in self.connection_genome.values():
                if cx.enabled and cx.key[1] == node_id:
                    required_cxs.add(cx.key)
        
        for cx in self.connection_genome.values():
            if cx.enabled and cx.key in required_cxs:
                params.append(cx.weight)
       
        # for n in self.node_genome.values():
        #     if n.key in required_nodes:
        #         params.append(n.bias)
        
        return params
    
    def get_params(self):
        """Returns a list of all parameters in the network."""
        params = self.get_cppn_params()
        params.extend(self.get_pre_params())
        params.extend(self.get_post_params())
        return params
    
    def get_post_params(self):
        """Returns a list of all parameters in the network."""
        params = []
        for layer in self.post_layers:
            params.extend(list(layer.parameters()))
        return params
    def get_pre_params(self):
        """Returns a list of all parameters in the network."""
        params = []
        for layer in self.pre_layers:
            params.extend(list(layer.parameters()))
        return params
    
    @property
    def num_params(self):
        cppn_len = len(self.get_cppn_params())
        for layer in self.pre_layers:
            for p in layer.parameters():
                cppn_len += p.numel()
        for layer in self.post_layers:
            for p in layer.parameters():
                cppn_len += p.numel()
        return cppn_len

    def get_named_params(self):
        """Returns a dict of all parameters in the network."""
        params = {}
        for cx in self.connection_genome.values():
            if cx.enabled:
                params[f"w_{cx.key}"] = cx.weight
        return params

    def prepare_optimizer(self, opt_class=torch.optim.Adam, lr=None, create_opt = False):
        """Prepares the optimizer."""
        assert self.config is not None, "Config is None."
        if lr is None:
            lr = self.config.sgd_learning_rate
        self.outputs = None # reset output
        
        # make a new computation graph
        for cx in self.connection_genome.values():
            if cx.enabled:
                cx.weight = torch.nn.Parameter(torch.tensor(cx.weight.detach().item(), requires_grad=True, dtype=self.config.dtype))
        # exit()
        if create_opt:
            self.optimizer = opt_class(self.get_params(), lr=lr)
            return self.optimizer
        else:
            return self.get_params()
    
    def initialize_connection_genome(self):
        """Initializes the connection genome."""
        assert self.config is not None, "Config is None."

        output_layer_idx = 2 # initialized to 2
        for layer_index in range(0, output_layer_idx+1):
            for layer_to_index in range(layer_index, output_layer_idx+1):
                if layer_index != layer_to_index:
                    for from_node in self.get_layer(layer_index):
                        for to_node in self.get_layer(layer_to_index):
                            new_cx = Connection(
                                (from_node.id, to_node.id), self.random_weight())
                            self.connection_genome[new_cx.key] = new_cx
                            if torch.rand(1)[0] > self.config.init_connection_probability:
                                new_cx.enabled = False
        
        # also connect inputs to outputs directly:
        if self.config.dense_init_connections and self.config.hidden_nodes_at_start > 0:
            for from_node in self.get_layer(0):
                for to_node in self.get_layer(output_layer_idx):
                    new_cx = Connection(
                        (from_node.id, to_node.id), self.random_weight())
                    self.connection_genome[new_cx.key] = new_cx
                    if torch.rand(1)[0] > self.config.init_connection_probability:
                        new_cx.enabled = False
        
    def serialize(self):
        # del type(self).constant_inputs
        assert self.config is not None, "Config is None."
        type(self).constant_inputs = None
        self.generator = None
        if self.outputs is not None:
            self.outputs = self.outputs.cpu().numpy().tolist() if\
                isinstance(self.outputs, torch.Tensor) else self.outputs
        for _, node in self.node_genome.items():
            node.serialize()
        for _, connection in self.connection_genome.items():
            connection.serialize()

        self.config.serialize()

    def deserialize(self):
        assert self.config is not None, "Config is None."
        for _, node in self.node_genome.items():
            node.deserialize()
        for _, connection in self.connection_genome.items():
            connection.deserialize()
        self.config.deserialize()
        
    def to_json(self):
        """Converts the CPPN to a json dict."""
        assert self.config is not None, "Config is None."

        self.serialize()
        img = json.dumps(self.outputs) if self.outputs is not None else None
        # make copies to keep the CPPN intact
        copy_of_nodes = copy.deepcopy(self.node_genome).items()
        copy_of_connections = copy.deepcopy(self.connection_genome).items()
        return {"id":self.id, "parents":self.parents, "fitness":self.fitness.item(), "novelty":self.novelty.item(), "node_genome": [n.to_json() for _,n in copy_of_nodes], "connection_genome":\
            [c.to_json() for _,c in copy_of_connections], "image": img, "selected": self.selected, "config": copy.deepcopy(self.config).to_json()}

    def from_json(self, json_dict):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        

        for key, value in json_dict.items():
            if key != "config":
                # TODO
                setattr(self, key, value)

        self.node_genome = {}
        self.connection_genome = {}
        for item in json_dict["node_genome"]:
            obj = json.loads(item, strict=False)
            self.node_genome[obj["id"]] = item
        for item in json_dict["connection_genome"]:
            obj = json.loads(item, strict=False)
            obj["key_"] = tuple(obj["key_"])
            self.connection_genome[obj["key_"]] = obj 

       # connections
        for key, cx in self.connection_genome.items():
            self.connection_genome[key] = Connection.create_from_json(cx) if\
                isinstance(cx, (dict,str)) else cx
            assert isinstance(self.connection_genome[key], Connection),\
                f"Connection is a {type(self.connection_genome[key])}: {self.connection_genome[key]}"

       # nodes
        for key, node in self.node_genome.items():
            self.node_genome[key] = Node.create_from_json(node) if\
                isinstance(node, (dict,str)) else node

            assert isinstance(self.node_genome[key], Node),\
                f"Node is a {type(self.node_genome[key])}: {self.node_genome[key]}"
        
        self.update_node_layers()
        assert self.config is not None, "Config is None."
        type(self).initialize_inputs(self.config.res_h, self.config.res_w,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.config.num_inputs,
                self.device)

    @staticmethod
    def create_from_json(json_dict, config=None, configClass=None):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        if config is None:
            assert configClass is not None, "Either config or configClass must be provided."
            json_dict["config"] = json_dict["config"].replace("cuda", "cpu").replace(":0", "")
            config = configClass.create_from_json(json_dict["config"])
        i = CPPN(config)
        i.from_json(json_dict)
        return i

    def random_weight(self):
        """Returns a random weight between -max_weight and max_weight."""
        assert self.config is not None, "Config is None."
        # return random_uniform(self.generator,
        #                       -self.config.max_weight,
        #                       self.config.max_weight,
        #                       grad=self.config.with_grad)
        return random_normal(self.generator,
                              0.0,
                              self.config.weight_init_std,
                              grad=self.config.with_grad).float()

    def enabled_connections(self):
        """Returns a yield of enabled connections."""
        for key, connection in self.connection_genome.items():
            if connection.enabled:
                yield connection
                
    def count_enabled_connections(self):
        """Returns the number of enabled connections."""
        return sum(1 for _ in self.enabled_connections())
    
    def count_nodes(self):
        """Returns the number of nodes."""
        return len(self.node_genome)
    
    def count_activation_functions(self):
        """Returns the number of unique activation functions."""
        return len(set([n.activation for n in self.node_genome.values()]))
    
    def mutate_activations(self, prob):
        """Mutates the activation functions of the nodes."""
        assert self.config is not None, "Config is None."
        if len(self.config.activations) == 1:
            return # no point in mutating if there is only one activation function

        eligible_nodes = list(self.hidden_nodes().values())
        if self.config.output_activation is None:
            eligible_nodes.extend(self.output_nodes().values())
        if self.config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes().values())
        for node in eligible_nodes:
            if random_uniform(self.generator,0,1) < prob:
                node.activation = choose_random_function(self.generator,self.config)
        self.outputs = None # reset the image

    def mutate_weights(self, prob):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """
        assert self.config is not None, "Config is None."
        R_delta = torch.rand(len(self.connection_genome.items()),generator=self.generator, device=self.generator.device)
        R_reset = torch.rand(len(self.connection_genome.items()),generator=self.generator, device=self.generator.device)

        for i, connection in enumerate(self.connection_genome.values()):
            if R_delta[i] < prob:
                delta = random_normal(self.generator, 0,
                                               self.config.weight_mutation_std)
                connection.weight = connection.weight + delta
            elif R_reset[i] < self.config.prob_weight_reinit:
                connection.weight = self.random_weight()

        # self.clamp_weights()
        self.outputs = None # reset the image

    def mutate_bias(self, prob):
        """
        """
        assert self.config is not None, "Config is None."
        R_delta = torch.rand(len(self.node_genome.items()),generator=self.generator, device=self.generator.device)
        R_reset = torch.rand(len(self.node_genome.items()),generator=self.generator, device=self.generator.device)

        for i, node in enumerate(self.node_genome.values()):
            if R_delta[i] < prob:
                delta = random_normal(self.generator, 0,
                                               self.config.bias_mutation_std)
                node.bias = node.bias + delta
            elif R_reset[i] < self.config.prob_weight_reinit:
                node.bias = torch.zeros_like(node.bias)

        self.clamp_weights()
        self.outputs = None # reset the image
        
    def depth(self):
        """Returns the depth of the network."""
        return len(self.get_layers())
    
    def width(self, agg=max):
        """Returns the width of the network."""
        return agg([len(layer) for layer in self.get_layers().values()])

    def mutate(self, rates=None):
        """Mutates the CPPN based on its config or the optionally provided rates."""
        self.fitness, self.adjusted_fitness, self.novelty = torch.tensor(-torch.inf,device=self.device), torch.tensor(-torch.inf,device=self.device), torch.tensor(-torch.inf,device=self.device) # new fitnesses after mutation
        assert self.config is not None, "Config is None."
        if rates is None:
            add_node = self.config.prob_add_node
            add_connection = self.config.prob_add_connection
            remove_node = self.config.prob_remove_node
            disable_connection = self.config.prob_disable_connection
            mutate_weights = self.config.prob_mutate_weight
            mutate_bias = self.config.prob_mutate_bias
            mutate_activations = self.config.prob_mutate_activation
        else:
            mutate_activations, mutate_weights, mutate_bias, add_connection, add_node, remove_node, disable_connection, weight_mutation_max, prob_reenable_connection = rates
        
        
        if random_uniform(self.generator,0.0,1.0) < add_node:
            self.add_node()
        if random_uniform(self.generator,0.0,1.0) < remove_node:
            self.remove_node()
        if random_uniform(self.generator,0.0,1.0) < add_connection:
            self.add_connection()
        if random_uniform(self.generator,0.0,1.0) < disable_connection:
            self.disable_connection()
        
        self.mutate_activations(mutate_activations)
        self.mutate_weights(mutate_weights)
        self.mutate_bias(mutate_bias)
        self.update_node_layers()
        # self.disable_invalid_connections()
        self.outputs = None # reset the image
        if hasattr(self, 'aot_fn'):
            del self.aot_fn # needs recompile
            
        

    def disable_invalid_connections(self):
        """Disables connections that are not compatible with the current configuration."""
        return # TODO: test, but there should never be invalid connections
        for key, connection in self.connection_genome.items():
            if connection.enabled:
                if not is_valid_connection(self.node_genome, connection.key, self.config):
                    connection.enabled = False

    def add_connection(self):
        """Adds a connection to the CPPN."""
        assert self.config is not None, "Config is None."
        
        for _ in range(20):  # try 20 times max
            [from_node, to_node] = random_choice(self.generator, 
                list(self.node_genome.values()), 2, replace=False)

            # look to see if this connection already exists
            existing_cx = self.connection_genome.get((from_node.id, to_node.id))

            # if it does exist and it is disabled, there is a chance to reenable
            if existing_cx is not None:
                if not existing_cx.enabled:
                    if random_uniform(self.generator) < self.config.prob_reenable_connection:
                        existing_cx.enabled = True # re-enable the connection
                break  # don't allow duplicates, don't enable more than one connection

            # else if it doesn't exist, check if it is valid
            if is_valid_connection(self.node_genome, (from_node.id, to_node.id), self.config):
                # valid connection, add
                new_cx = Connection((from_node.id, to_node.id), self.random_weight())
                assert new_cx.key not in self.connection_genome.keys(),\
                    "CX already exists: {}".format(new_cx.key)
                self.connection_genome[new_cx.key] = new_cx
                self.update_node_layers()
                break # found a valid connection, don't add more than one

            # else failed to find a valid connection, don't add and try again
        self.outputs = None # reset the image

    def add_node(self):
        """Adds a node to the CPPN.
            Looks for an eligible connection to split, add the node in the middle
            of the connection.
        """
        # only add nodes in the middle of non-recurrent connections
        eligible_cxs = [
            cx for k, cx in self.connection_genome.items() if not cx.is_recurrent]

        if len(eligible_cxs) == 0:
            return # there are no eligible connections, don't add a node

        # choose a random eligible connection
        old_connection = random_choice(self.generator, eligible_cxs,1,replace=False)[0]

        # create the new node
        new_node = Node(self.get_new_node_id(), choose_random_function(self.generator, self.config),
                        NodeType.HIDDEN, 999, self.config.node_agg, device=self.device, grad=self.config.with_grad)
        
        assert new_node.id not in self.node_genome.keys(),\
            "Node ID already exists: {}".format(new_node.id)
            
        self.node_genome[new_node.id] =  new_node # add a new node between two nodes

        # disable old connection
        old_connection.enabled = False

        # The connection between the first node in the chain and the
        # new node is given a weight of one and the connection between
        # the new node and the last node in the chain
        # is given the same weight as the connection being split
        new_cx_1 = Connection(
            (old_connection.key[0], new_node.id), torch.tensor(1.0, device=self.device,dtype=self.config.dtype, requires_grad=self.config.with_grad))
        assert new_cx_1.key not in self.connection_genome.keys()
        self.connection_genome[new_cx_1.key] = new_cx_1

        new_cx_2 = Connection((new_node.id, old_connection.key[1]),
            old_connection.weight)
        assert new_cx_2.key not in self.connection_genome.keys()
        self.connection_genome[new_cx_2.key] = new_cx_2

        self.update_node_layers() # update the layers of the nodes
        self.outputs = None # reset the image
        
    def remove_node(self):
        """Removes a node from the CPPN.
            Only hidden nodes are eligible to be removed.
        """

        hidden = self.hidden_nodes().values()
        if len(hidden) == 0:
            return # no eligible nodes, don't remove a node

        # choose a random node
        node_id_to_remove = random_choice(self.generator, [n.id for n in hidden], 1, False)[0]

        for key, cx in list(self.connection_genome.items())[::-1]:
            if node_id_to_remove in cx.key:
                del self.connection_genome[key]
        for key, node in list(self.node_genome.items())[::-1]:
            if node.id == node_id_to_remove:
                del self.node_genome[key]
                break

        self.update_node_layers()
        self.disable_invalid_connections()
        self.outputs = None # reset the image

    def prune(self):
        removed = 0
        for cx in list(self.connection_genome.values())[::-1]:
            if abs(cx.weight )< self.config.prune_threshold:
                del self.connection_genome[cx.key]
                removed += 1
        for _ in range(self.config.min_pruned - removed):
            min_weight_key = min(self.connection_genome, key=lambda k: self.connection_genome[k].weight)
            removed += 1
            del self.connection_genome[min_weight_key]
       
        # remove nodes with no connections
        all_keys = []
        all_keys.extend([cx.key[0] for cx in self.connection_genome.values()])
        all_keys.extend([cx.key[1] for cx in self.connection_genome.values()])

        for node in list(self.node_genome.values())[::-1]:
            if node.id not in all_keys:
                del self.node_genome[node.id]
        
        # print("Pruned {} connections".format(removed))
        
        self.update_node_layers()
        self.disable_invalid_connections()
        self.outputs = None # reset the image

    
    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections())
        if len(eligible_cxs) < 1:
            return
        cx = random_choice(self.generator, eligible_cxs, 1, False)[0]
        cx.enabled = False
        self.outputs = None # reset the image

    def update_node_layers(self):
        """Update the node layers."""
        layers = feed_forward_layers(self)
        for _, node in self.input_nodes().items():
            node.layer = 0
        for layer_index, layer in enumerate(layers):
            for node_id in layer:
                node = find_node_with_id(self.node_genome, node_id)
                node.layer = layer_index + 1

    def input_nodes(self) -> dict:
        """Returns a dict of all input nodes."""
        return {n.id: n for n in self.node_genome.values() if n.type == NodeType.INPUT}

    def output_nodes(self) -> dict:
        """Returns a dict of all output nodes."""
        return {n.id: n for n in self.node_genome.values() if n.type == NodeType.OUTPUT}

    def hidden_nodes(self) -> dict:
        """Returns a dict of all hidden nodes."""
        return {n.id: n for n in self.node_genome.values() if n.type == NodeType.HIDDEN}

    def set_inputs(self, inputs):
        """Sets the input neurons outputs to the input values."""
        assert self.config is not None, "Config is None."
        if self.config.use_radial_distance:
            # d = sqrt(x^2 + y^2)
            inputs.append(torch.sqrt(inputs[0]**2 + inputs[1]**2))
        if self.config.use_input_bias:
            inputs.append(torch.tensor(1.0,device=self.device))  # bias = 1.0
            
        assert len(inputs) == len(self.input_nodes()), f"Wrong number of inputs {len(inputs)} != {len(self.input_nodes())}"
        
        for inp, node in zip(inputs, self.input_nodes().values()):
            # inputs are first N nodes
            node.sum_inputs = inp
            node.outputs = node.activation(inp)

    def get_layer(self, layer_index):
        """Returns a list of nodes in the given layer."""
        for _, node in self.node_genome.items():
            if node.layer == layer_index:
                yield node

    def get_layers(self) -> dict:
        """Returns a dictionary of lists of nodes in each layer."""
        layers = {}
        for _, node in self.node_genome.items():
            if node.layer not in layers:
                layers[node.layer] = []
            layers[node.layer].append(node)
        return layers

    def clamp_weights(self):
        """Clamps all weights to the range [-max_weight, max_weight]."""
        assert self.config is not None, "Config is None."
        if not self.config.get("clamp_weights", True):
            return
            
        for _, cx in self.connection_genome.items():
            if cx.weight < self.config.weight_threshold and cx.weight >\
                 -self.config.weight_threshold:
                cx.weight = torch.tensor(0.0, device=self.device, requires_grad=cx.weight.requires_grad)
            if not isinstance(cx.weight, torch.Tensor):
                cx.weight = torch.tensor(cx.weight, device=self.device, requires_grad=cx.weight.requires_grad)
            cx.weight = torch.clamp(cx.weight, min=-self.config.max_weight, max=self.config.max_weight)
            

    def reset_activations(self, parallel=True):
        """Resets all node activations to zero."""
        assert self.config is not None, "Config is None."
        h,w = self.config.res_h, self.config.res_w
        h,w = h//2**self.config.num_upsamples, w//2**self.config.num_upsamples
        if parallel:
            for _, node in self.node_genome.items():
                node.sum_inputs = torch.zeros((1, h, w), device=self.device)
                node.outputs = torch.zeros((1, h, w), device=self.device)
        else:
            for _, node in self.node_genome.items():
                node.sum_inputs = torch.zeros(1, device=self.device)
                node.outputs = torch.zeros(1, device=self.device)
                
    def eval(self, inputs):
        """Evaluates the CPPN."""
        raise NotImplementedError("Use forward() instead.")

    def forward_(self, inputs=None, extra_inputs=None, channel_first=True):
        # TODO: channel_first is ignored for now
        
        if self.device!=inputs.device:
            self.to(inputs.device) # breaks computation graph
        
        assert self.config is not None, "Config is None."
        
        batch_size = 1 if extra_inputs is None else extra_inputs.shape[0]
        res_h, res_w = self.config.res_h, self.config.res_w
        res_h, res_w = res_h//2**self.config.num_upsamples, res_w//2**self.config.num_upsamples
        
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert isinstance(res_h, int), "Res h must be an integer."
        assert isinstance(res_w, int), "Res w must be an integer."
        
        if inputs is None:
            inputs = type(self).constant_inputs
        for p_layer in self.pre_layers:
            inputs = inputs.permute(2,0,1)
            inputs = p_layer(inputs)
            inputs = torch.relu(inputs)
            inputs = inputs.permute(1, 2, 0)

        # reset the activations to 0 before evaluating
        self.reset_activations()
        
        # get layers
        layers = feed_forward_layers(self) 
        layers.insert(0, self.input_nodes().keys()) # add input nodes as first layer
        
        # iterate over layers
        for layer in layers:
            Xs, Ws, nodes, Fns = [], [], [], []
            for node_index, node_id in enumerate(layer):
                # iterate over nodes in layer
                node = self.node_genome[node_id] # the current node
                
                if node.type == NodeType.INPUT:
                    # initialize the node's sum
                    X = inputs[:,:,node_index].repeat(batch_size, 1, 1, 1) # (batch_size, cx, res_h, res_w)
                    weights = torch.ones((1), dtype=self.config.dtype, device=self.device)
                else:
                    # find incoming connections and activate
                    X, weights = get_incoming_connections_weights(self, node)
                    # X shape = (batch_size, num_incoming, res_h, res_w)
                    if X is None:
                        X = torch.zeros((batch_size, 1, res_h, res_w), dtype=self.config.dtype, device=self.device)
                    if weights is None:
                        weights = torch.ones((1), dtype=self.config.dtype, device=self.device)

                if self.config.activation_mode == 'node':
                    node.activate(X, weights) # naive
                elif self.config.activation_mode == 'layer':
                    # group by function for efficiency
                    Xs.append(X)
                    Ws.append(weights)
                    nodes.append(node)
                else:
                    raise ValueError(f"Unknown activation mode {self.config.activation_mode}")
                
            if self.config.activation_mode == 'layer':
                activate_layer(Xs, Ws, {n.id:n for n in nodes}, self.config.node_agg)
            
        # collect outputs from the last layer
        sorted_o = sorted(self.output_nodes().values(), key=lambda x: x.key, reverse=True)
        outputs = torch.stack([node.outputs for node in sorted_o], dim=1)
        assert str(outputs.device) == str(self.device), f"Output is on {outputs.device}, should be {self.device}"

        self.outputs = outputs
        for layer in self.post_layers:
            if isinstance(layer, torch.nn.Linear):
                self.outputs = self.outputs.reshape(self.outputs.shape[0], -1)
            self.outputs = layer(self.outputs)
            # self.outputs = torch.relu(self.outputs)
            if isinstance(layer, torch.nn.Linear):
                self.outputs = self.outputs.reshape(self.outputs.shape[0], len(self.config.color_mode), self.config.res_h, self.config.res_w)
                
                
        assert str(self.outputs.device )== str(self.device), f"Output is on {self.outputs.device}, should be {self.device}"
        assert self.outputs.dtype == torch.float32, f"Output is {self.outputs.dtype}, should be float32"
        return self.outputs
    
    def forward(self, inputs=None, extra_inputs=None, no_aot=True, channel_first=True):
        """Feeds forward the network. A wrapper for forward_() that allows for AOT compilation."""
        
        assert self.config is not None, "Config is None."
        if self.config.with_grad and not no_aot:
            if not hasattr(self, 'aot_fn'):
                def f(x0, x1):
                    return self.forward_(inputs=x0, extra_inputs=x1, channel_first=channel_first) 
                def fw(f, inps):
                    return f
                def bw(f, inps):
                    return f
                
                self.aot_fn = torch.compile(f)
            
            return self.aot_fn(inputs, extra_inputs)  
        else:  
            return self.forward_(inputs=inputs, extra_inputs=extra_inputs, channel_first=channel_first)

    def backward(self, loss:torch.Tensor,retain_graph=False):
        """Backpropagates the error through the network."""
        assert self.config is not None
        assert self.config.with_grad, "Cannot backpropagate without gradients. Set config.with_grad=True."
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        self.outputs = None # new image
    
    def discard_grads(self):
        for _, cx in self.connection_genome.items():
            # check nan
            if torch.isnan(cx.weight).any():
                # TODO: why NaN?
                cx.weight = torch.tensor(0, device=self.device)
            else:
                cx.weight = torch.tensor(cx.weight.detach().item(), device=self.device, dtype=self.config.dtype)
        for _, node in self.node_genome.items():
            node.bias = torch.tensor(node.bias.detach().item(), device=self.device, dtype=self.config.dtype)
        self.reset_activations()
        self.outputs = None # new image
        self.fitness= torch.tensor(self.fitness.detach().item(), device=self.device)
        if hasattr(self, 'optimizer'):
            del self.optimizer
            self.optimizer = None
        if hasattr(self, 'aot_fn'):
            del self.aot_fn
        
    
    def genetic_difference(self, other) -> float:
        # only enabled connections, sorted by innovation id
        this_cxs = sorted(self.enabled_connections(),
                          key=lambda c: c.key)
        other_cxs = sorted(other.enabled_connections(),
                           key=lambda c: c.key)

        N = max(len(this_cxs), len(other_cxs))
        other_innovation = [c.key for c in other_cxs]

        # number of excess connections
        n_excess = len(get_excess_connections(this_cxs, other_innovation))
        # number of disjoint connections
        n_disjoint = len(get_disjoint_connections(this_cxs, other_innovation))

        # matching connections
        this_matching, other_matching = get_matching_connections(
            this_cxs, other_cxs)
        
        difference_of_matching_weights = [
            abs(o_cx.weight.item()-t_cx.weight.item()) for o_cx, t_cx in zip(other_matching, this_matching)]
        # difference_of_matching_weights = torch.stack(difference_of_matching_weights)
        
        if(len(difference_of_matching_weights) == 0):
            difference_of_matching_weights = 0
        else:
            difference_of_matching_weights = sum(difference_of_matching_weights) / len(difference_of_matching_weights)

        # Furthermore, the compatibility distance function
        # includes an additional argument that counts how many
        # activation functions differ between the two individuals
        n_different_fns = 0
        for t_node, o_node in zip(self.node_genome.values(), other.node_genome.values()):
            if(t_node.activation.__name__ != o_node.activation.__name__):
                n_different_fns += 1

        # can normalize by size of network (from Ken's paper)
        if(N > 0):
            n_excess /= N
            n_disjoint /= N

        # weight (values from Ken)
        n_excess *= 1
        n_disjoint *= 1
        difference_of_matching_weights *= .4
        n_different_fns *= 1
        
        difference = sum([n_excess,
                          n_disjoint,
                          difference_of_matching_weights,
                          n_different_fns])
        if torch.isnan(torch.tensor(difference)):
            difference = 0

        return difference

    def species_comparision(self, other, threshold) -> bool:
        # returns whether other is the same species as self
        return self.genetic_difference(other) < threshold


    def crossover(self, other):
        """ Configure a new genome by crossover from two parent genomes. """
        
        # TODO: this may mess up the order of the outputs, need to check
        
        child = CPPN(self.config, {}, {}) # create an empty child genome
        assert self.fitness is not None, "Parent 1 has no fitness"
        assert other.fitness is not None, "Parent 2 has no fitness"
        # determine which parent is more fit
        if self.fitness > other.fitness:
            parent1, parent2 = self, other
        elif self.fitness < other.fitness:
            parent1, parent2 = other, self
        else:
            # fitness same, choose randomly
            if torch.rand(1,generator=self.generator)[0] < 0.5:
                parent1, parent2 = self, other
            else:
                parent1, parent2 = other, self

        # Inherit connection genes
        for key, cx1 in parent1.connection_genome.items():
            cx2 = parent2.connection_genome.get(key)
            if cx2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                child.connection_genome[key] = cx1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                child.connection_genome[key] = cx1.crossover(cx2)

        # Inherit node genes
        for key, node1 in parent1.node_genome.items():
            node2 = parent2.node_genome.get(key)
            assert key not in child.node_genome
            if node2 is None:
                # Extra gene: copy from the fittest parent
                child.node_genome[key] = node1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                child.node_genome[key] = node1.crossover(node2)

        child.parents = (parent1.id, parent2.id)
        child.update_node_layers()
        
        return child

    def save(self, path):
        copy = self.clone()
        json_ = copy.to_json()
        assert os.path.exists(path), f"Path {path} does not exist"
        with open(path, 'w') as f:
            json.dump(json_, f, indent=4)
            
    def compress(self, path):
        """Save the genome as a compressed numpy file."""
        self.cpu()
        self.discard_grads()
        weight_arr = np.zeros((len(list(self.enabled_connections()))), dtype=np.float32)
        id_arr = np.zeros((len(list(self.enabled_connections())), 2), dtype=np.int8)
        act_arr = np.zeros((len(list(self.enabled_connections())), 2), dtype=np.int8)
        types_arr = np.zeros((len(list(self.enabled_connections())), 2), dtype=np.int8)
        activation_idx = {str(act): i for i, act in enumerate(self.config.activations)}
        version = np.array(self.config.version, dtype=np.int8)
        
        for i, cx in enumerate(self.enabled_connections()):
            in_node, out_node = cx.key
            id_arr[i, 0] = in_node
            id_arr[i, 1] = out_node
            act_arr[i, 0] = activation_idx[str(self.node_genome[in_node].activation)]
            act_arr[i, 1] = activation_idx[str(self.node_genome[out_node].activation)]
            types_arr[i, 0] = int(self.node_genome[in_node].type)
            types_arr[i, 1] = int(self.node_genome[out_node].type)
            weight_arr[i] = cx.weight
        with open(path, 'wb') as f:
            np.savez_compressed(f, version=version, weight_arr=weight_arr, id_arr=id_arr, act_arr=act_arr, types_arr=types_arr)
        
    def decompress(self, path):
        """Load the genome from a compressed numpy file."""
        assert os.path.exists(path), f"Path {path} does not exist"
        loaded = np.load(path)
        version = loaded['version']
        weight_arr = loaded['weight_arr']
        id_arr = loaded['id_arr']
        act_arr = loaded['act_arr']
        types_arr = loaded['types_arr']
        idx_activation = {i: act for i, act in enumerate(self.config.activations)}
        
        if version.tolist() != self.config.version:
            logging.warn(f"Version mismatch. Expected {self.config.version}, got {version.tolist()}")
            logging.warn("Uncompressed file may not be compatible with current configuration.")
            logging.warn(f"Use CPPN version {version.tolist()} for compatibility.")
            
        self.node_genome = {}
        nodes = []
        self.connection_genome = {}
        cx_id = 0
        for ids, acts, weight in zip(id_arr, act_arr, weight_arr):
            in_node, out_node = ids
            act_in, act_out = acts
            if in_node not in self.node_genome:
                in_node = Node(in_node, idx_activation[act_in], types_arr[cx_id][0], device=self.device, grad=self.config.with_grad)
            if out_node not in self.node_genome:
                self.node_genome[out_node] = Node(out_node, idx_activation[act_out], types_arr[cx_id][1], self.config.node_agg, device=self.device, grad=self.config.with_grad)
            self.connection_genome[(in_node, out_node)] = Connection((in_node, out_node), torch.tensor(weight, dtype=self.config.dtype, device=self.device, requires_grad=self.config.with_grad))
            cx_id += 1

        self.update_node_layers()
        
    


    def clone(self, deepcopy=True, cpu=False, new_id=False):
        """ Create a copy of this genome. """
        if hasattr(self, 'aot_fn'):
            # can't clone AOT functions
            del self.aot_fn

        if hasattr(self, 'generator'):
            generator_state = self.generator.get_state()
            del self.generator
        else:
            generator_state = torch.Generator(device=self.device).get_state()
        
        out_child = None
        
        id = self.id if (not new_id) else type(self).get_id()
        if deepcopy:
            if self.config.with_grad:
                self.discard_grads()

            child = copy.deepcopy(self)
            out_child = child
        else:
            child = type(self)(self.config, {}, {})
            child.connection_genome = {key: cx.copy() for key, cx in self.connection_genome.items()}        
            child.node_genome = {key: node.copy() for key, node in self.node_genome.items()}
            child.update_node_layers()
            for cx in child.connection_genome.values():
                cx.weight = cx.weight.detach()
                cx.weight = torch.tensor(cx.weight.item(), dtype=self.config.dtype) # require prepare_optimizer() again

            out_child = child
        
        self.generator = torch.Generator(device=self.device)
        try:
            self.generator.set_state(generator_state)
        except RuntimeError:
            ...

        out_child.set_id(id)
        out_child.age = 0
        
        if cpu:
            out_child.cpu()

        out_child.configure_generator()

        
        for n in self.node_genome.values():
            n.bias = torch.tensor(n.bias.item(), dtype=self.config.dtype, device=self.device)
        for n in out_child.node_genome.values():
            n.bias = torch.tensor(n.bias.item(), dtype=self.config.dtype, device=out_child.device)
        return out_child
    
    

    def layer_to_str(self, layer, i):
        if isinstance(layer, Conv2d):
            return f'CONV {i}\n{layer.kernel_size}x{layer.in_channels}'
        elif isinstance(layer, torch.nn.MaxPool2d):
            return f'POOL {i}\n{layer.kernel_size}x{layer.in_channels}'
        elif isinstance(layer, torch.nn.MultiheadAttention):
            return f'ATTN {i}\n{layer.num_heads} heads'
        elif isinstance(layer, torch.nn.Sequential) and isinstance(layer[0], torch.nn.Upsample):
            return f'UPSAMPLE {i}\nX{layer[0].scale_factor}'
        elif isinstance(layer, torch.nn.Upsample):
            return f'UPSAMPLE {i}\nX{layer.scale_factor}'
        elif isinstance(layer, torch.nn.Flatten):
            return f'FLATTEN {i}'
        elif isinstance(layer, torch.nn.Unflatten):
            return f'UNFLATTEN {i}'
        elif isinstance(layer, torch.nn.Linear):
            return f'LINEAR {i}'
        elif isinstance(layer, Block):
            return f'RESNET BLOCK {i}'

    def to_nx(self):
        import networkx as nx
        G = nx.DiGraph()
        for i, layer in enumerate(self.pre_layers):
            G.add_node(self.layer_to_str(layer, i))
         
        for i, layer in enumerate(self.post_layers):
            G.add_node(self.layer_to_str(layer, i))
            
        for i, layer in enumerate(self.pre_layers[:-1]):
            G.add_edge(self.layer_to_str(layer, i), self.layer_to_str(self.pre_layers[i+1], i+1))
            
        for i, layer in enumerate(self.post_layers[:-1]):
            G.add_edge(self.layer_to_str(layer, i), self.layer_to_str(self.post_layers[i+1], i+1))
        
        for n in self.node_genome.values():
            G.add_node(n.key, type=n.type.name, fn=n.activation)
        for c in self.connection_genome.values():
            if c.enabled:
                G.add_edge(c.key[0], c.key[1], weight=c.weight.item())
                
        if len(self.pre_layers) > 0:
            last_layer = self.pre_layers[-1]
            last_layer_i = len(self.pre_layers)-1
            for n in self.input_nodes().values():
                G.add_edge(self.layer_to_str(last_layer, last_layer_i), n.key)
                
        if len(self.post_layers) > 0:
            first_layer = self.post_layers[0]
            first_layer_i = 0
            for n in self.output_nodes().values():
                G.add_edge(n.key, self.layer_to_str(first_layer, first_layer_i))
        return G
    
    
    def to_networkx(self):
        self.update_node_layers()
        G = nx.DiGraph()
        used_nodes = set()
        for key, cx in self.connection_genome.items():
            if cx.enabled:
                G.add_edge(key[0], key[1], weight=cx.weight.item())
                used_nodes.add(key[0])
                used_nodes.add(key[1])
                
        for key, node in self.node_genome.items():
            if key in used_nodes:
                G.add_node(key, activation=node.activation.__name__)
        
        return G

    def draw_nx(self, size=(10,20), show=True):
        import matplotlib.pyplot as plt
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        fig = plt.figure(figsize=size)
        G = self.to_nx()
        pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw_networkx(
            G,
            with_labels=True,
            pos=pos,
            labels={n:f"{n}\n{self.node_genome[n].activation.__name__[:4]}\n{1}xHxW"if n in self.node_genome else n
                    for n in G.nodes(data=False) },
            node_size=800,
            font_size=6,
            node_shape='s',
            node_color=['lightsteelblue' if n in self.node_genome else 'lightgreen' for n in G.nodes()  ]
            )
        plt.annotate('# params: ' + str(self.num_params), xy=(1.0, 1.0), xycoords='axes fraction', fontsize=12, ha='right', va='top')
        if show:
            plt.show()


    def to(self, device):
        self.device = device
        for pre in self.pre_layers:
            pre.to(device)
        for post in self.post_layers:
            post.to(device)
        for key, node in self.node_genome.items():
            node.to(device)
        for key, cx in self.connection_genome.items():
            cx.to(device)
                
    def cpu(self):
        '''Move all tensors to CPU'''
        if self.device == torch.device('cpu'):
            return
        self.device = torch.device('cpu')
        for key, node in self.node_genome.items():
            node.to_cpu()
        for key, cx in self.connection_genome.items():
            cx.to_cpu()
        self.fitness = self.fitness.cpu()
        if self.outputs is not None:
            self.outputs = self.outputs.cpu()
        self.adjusted_fitness = self.adjusted_fitness.cpu()
        self.novelty = self.novelty.cpu()

        for layer in self.pre_layers:
            layer.cpu()
        for layer in self.post_layers:
            layer.cpu()

    def cuda(self, device='cuda:0'):
        '''Move all tensors to CUDA'''
        if self.device == torch.device(device):
            return
        self.device = torch.device(device)
        for key, node in self.node_genome.items():
            node.to_cuda(device)
        for key, cx in self.connection_genome.items():
            cx.to_cuda(device)
        self.fitness = self.fitness.cuda(device)
        if self.outputs is not None:
            self.outputs = self.outputs.cuda(device)
        self.adjusted_fitness = self.adjusted_fitness.cuda(device)
        self.novelty = self.novelty.cuda(device)
        
        for layer in self.pre_layers:
            layer.cuda(device)
        for layer in self.post_layers:
            layer.cuda(device)