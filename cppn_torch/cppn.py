"""Contains the CPPN, Node, and Connection classes."""
import copy
from enum import IntEnum
from itertools import count
import math
import json
import os
import random
from cppn_torch.graph_util import activate_layer
import torch
from torch.nn import ConvTranspose2d, Conv2d
import copy
from copy import Error, _deepcopy_dispatch, _deepcopy_atomic, _keep_alive, _reconstruct
from copyreg import dispatch_table
import networkx as nx
import logging
from torch import nn
from functorch.compile import compiled_function, draw_graph, aot_function
from cppn_torch.activation_functions import IdentityActivation
import cppn_torch.activation_functions as af
from cppn_torch.graph_util import *
# from cppn_torch.config import CPPNConfig as Config
from cppn_torch.gene import * 
from cppn_torch.util import upscale_conv2d, random_choice, random_normal, random_uniform, gaussian_blur

from torchviz import make_dot


dtype = torch.float32

class CPPN(nn.Module):
    """A CPPN Object with Nodes and Connections."""

    constant_inputs = None # (res_h, res_w, n_inputs)
    # constant_inputs = torch.zeros((0, 0, 0), dtype=torch.float32,requires_grad=False) # (res_h, res_w, n_inputs)
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

    @staticmethod
    def initialize_inputs_from_config(config):
        return CPPN.initialize_inputs( 
                                        config.res_h, 
                                        config.res_w,
                                        config.use_radial_distance,
                                        config.use_input_bias,
                                        config.num_inputs,
                                        config.device,
                                        config.coord_range,
                                        CPPN,
                                        dtype
                                        )

    @staticmethod
    def get_id():
        __class__.current_id += 1
        return __class__.current_id - 1
    
    @staticmethod
    def create_from_json(json_dict, config=None, configClass=None, CPPNClass=None):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        if config is None:
            assert configClass is not None, "Either config or configClass must be provided."
            json_dict["config"] = json_dict["config"].replace("cuda", "cpu").replace(":0", "")
            config = configClass.create_from_json(json_dict["config"])
        if CPPNClass is None:
            CPPNClass = CPPN
        i = CPPNClass(config)
        i.from_json(json_dict)
        return i

    
    def __init__(self, config, nodes = None, connections = None) -> None:
        """Initialize a CPPN."""
        super().__init__()
        self.device = config.device
        self.outputs = None
        self.normalize_outputs = config.normalize_outputs # todo unused in CPPN
        self.node_genome = nn.ModuleDict()
        self.connection_genome = nn.ModuleDict()
        self.id = type(self).get_id()
        
        self.reconfig(config, nodes, connections)
        
        self.parents = (-1, -1)
        self.age = 0
        self.lineage = []
    

         
    @property
    def params(self):
        return self.get_params()
    
    @property
    def num_params(self):
        cppn_len = len(self.get_cppn_params())
        return cppn_len
    
   
    def reconfig(self, config = None, nodes = None, connections = None):
        self.device = config.device
            
        if self.device is None:
            raise ValueError("device is None") 

        self.n_outputs = len(config.color_mode) # RGB: 3, HSV: 3, L: 1
        self.n_in_nodes = config.num_inputs
       
        if nodes is None:
            self.initialize_node_genome(config)
        else:
            assert isinstance(nodes, nn.ModuleDict)
            self.node_genome = nodes
        if connections is None:
            self.initialize_connection_genome(config)
        else:
            assert isinstance(connections, nn.ModuleDict)
            self.connection_genome = connections
        
        self.disable_invalid_connections(config)
        
        self.sgd_lr = config.sgd_learning_rate
        
        self.output_blur = config.output_blur
        
        self.color_mode = config.color_mode
        
        self.graph = {}
        

    def get_new_node_id(self):
        """Returns a new node id`."""
        if type(self).node_indexer is None:
            if self.node_genome == {}:
                return 0
            if self.node_genome is not None:
                type(self).node_indexer = count(max([int(k) for k in self.node_genome.keys()]) + 1)
            else:
                type(self).node_indexer = count(max([int(k) for k in self.node_genome.keys()]) + 1)

        new_id = str(next(type(self).node_indexer))
        assert new_id not in self.node_genome.keys()
        return new_id
    
    def set_id(self, id):
        self.id = id
        
    def vis(self, fname='cppn_graph'):
        """Visualize the CPPN."""
        make_dot(self.outputs, show_attrs=True, show_saved=True).render(fname, format="pdf")
        
    def initialize_node_genome(self, config):
        """Initializes the node genome."""
        n_in = self.n_in_nodes
        n_out = self.n_outputs  
        n_hidden = config.hidden_nodes_at_start
       
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        
        output_layer_idx = 1+len(n_hidden) if sum(n_hidden) > 0 else 1
        
        # Input nodes:
        for idx in range(n_in):
            fn = IdentityActivation if not config.allow_input_activation_mutation else random_choice(config.activations)
            new_node = Node(f"{-(1+idx)}", fn, NodeType.INPUT, 0, config.node_agg, self.device, grad=config.with_grad)
            self.node_genome[new_node.key] = new_node
            
        # Output nodes:
        for idx in range(n_in, n_in + n_out):
            if config.output_activation is None:
                output_fn = random_choice(config.activations)
            else:
                output_fn = config.output_activation
            new_node = Node(f"{-(1+idx)}", output_fn, NodeType.OUTPUT, output_layer_idx, config.node_agg, self.device, grad=config.with_grad)
            self.node_genome[new_node.key] = new_node
        
        # the rest are hidden:
        for hidden_layer_idx, layer_size in enumerate(n_hidden):
            for idx in range(layer_size):
                new_node = Node(f"{self.get_new_node_id()}", random_choice(config.activations),
                                NodeType.HIDDEN, 1+hidden_layer_idx, config.node_agg, self.device, grad=config.with_grad)
                self.node_genome[new_node.key] = new_node

    def initialize_connection_genome(self, config):
        """Initializes the connection genome."""
        n_hidden = config.hidden_nodes_at_start
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        output_layer_idx = 1+len(n_hidden)

        std = config.weight_init_std

        # connect all layers to the next layer
        for layer_idx in range(output_layer_idx):
            for from_node in self.get_layer(layer_idx):
                for to_node in self.get_layer(layer_idx+1):
                    new_cx = Connection(
                        f"{from_node.id},{to_node.id}", self.random_weight(False, std))
                    self.connection_genome[new_cx.key] = new_cx
                    if torch.rand(1)[0] > config.init_connection_probability:
                        new_cx.enabled = False
        
        # also connect inputs to outputs directly:
        if config.dense_init_connections and sum(n_hidden) > 0:
            for from_node in self.get_layer(0):
                for to_node in self.get_layer(output_layer_idx):
                    new_cx = Connection(
                         f"{from_node.id},{to_node.id}", self.random_weight(False, std))
                    self.connection_genome[new_cx.key] = new_cx
                    if torch.rand(1)[0] > config.init_connection_probability:
                        new_cx.enabled = False
        

    def get_params(self):
        """Returns a list of all parameters in the network."""
        params = []
        required_nodes = required_for_output(*get_ids_from_individual(self))
        
        # cxs that end at a required node
        required_cxs = set()
        for node_id in required_nodes:
            for cx in self.connection_genome.values():
                key = cx.key.split(',')
                if cx.enabled and key[1] == node_id and key[0] in required_nodes:
                    required_cxs.add(cx.key)
        
        for cx in self.connection_genome.values():
            if cx.enabled and cx.key in required_cxs:
                params.append(cx.weight)
       
        for n in self.node_genome.values():
            if n.key in required_nodes:
                params.extend(n.params())
        
        return params
    
    
    def prepare_optimizer(self, opt_class=torch.optim.Adam, lr=None, create_opt=False):
        """Prepares the optimizer."""
        if lr is None:
            lr = self.sgd_lr
        self.outputs = None # reset output
        
        # make a new computation graph
        for cx in self.connection_genome.values():
            if cx.enabled:
                cx.weight = torch.nn.Parameter(torch.tensor(cx.weight.detach().item(), requires_grad=True, dtype=dtype, device = self.device))
        # exit()
        if create_opt:
            self.optimizer = opt_class(self.get_params(), lr=lr)
            return self.optimizer
        else:
            return self.get_params()
    
        
    def serialize(self):
        return
        # del type(self).constant_inputs
        type(self).constant_inputs = None
        if self.outputs is not None:
            self.outputs = self.outputs.detach().cpu().numpy().tolist() if\
                isinstance(self.outputs, torch.Tensor) else self.outputs
        for _, node in self.node_genome.items():
            node.serialize()
        for _, connection in self.connection_genome.items():
            connection.serialize()
            
        if isinstance(self.sgd_lr, torch.Tensor):
            self.sgd_lr = self.sgd_lr.item()

    def deserialize(self):
        for _, node in self.node_genome.items():
            node.deserialize()
        for _, connection in self.connection_genome.items():
            connection.deserialize()
        self.config.deserialize()
        
    def to_json(self):
        """Converts the CPPN to a json dict."""
        return {}
        self.serialize()
        # img = json.dumps(self.outputs) if self.outputs is not None else None
        # make copies to keep the CPPN intact
        copy_of_nodes = copy.deepcopy(self.node_genome).items()
        copy_of_connections = copy.deepcopy(self.connection_genome).items()
        return {"id":self.id, "parents":self.parents, "node_genome": [n.to_json() for _,n in copy_of_nodes], "connection_genome":\
            [c.to_json() for _,c in copy_of_connections], "lineage": self.lineage, "sgd_lr": self.sgd_lr}

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
        

    def random_weight(self, grad=False, std=1.0, mean=0.0):
        """Returns a random weight between -max_weight and max_weight."""
        return torch.randn(1, device=self.device, requires_grad=grad)[0] * std + mean


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
    
     
    def depth(self):
        """Returns the depth of the network."""
        return len(self.get_layers())
    
    
    def width(self, agg=max):
        """Returns the width of the network."""
        return agg([len(layer) for layer in self.get_layers().values()])
    
    
    def input_nodes(self) -> dict:
        """Returns a dict of all input nodes."""
        return {n.id: n for n in self.node_genome.values() if n.type == NodeType.INPUT}


    def output_nodes(self) -> dict:
        """Returns a dict of all output nodes."""
        return {n.id: n for n in self.node_genome.values() if n.type == NodeType.OUTPUT}


    def hidden_nodes(self) -> dict:
        """Returns a dict of all hidden nodes."""
        return {n.id: n for n in self.node_genome.values() if n.type == NodeType.HIDDEN}
    
    def get_named_params(self):
        """Returns a dict of all parameters in the network."""
        params = {}
        for cx in self.connection_genome.values():
            if cx.enabled:
                params[f"w_{cx.key}"] = cx.weight
        return params
    
    def named_parameters(self):
        """In torch format"""
        return self.get_named_params()

    
    
    def mutate_activations(self, prob, config):
        """Mutates the activation functions of the nodes."""
        if len(config.activations) == 1:
            return # no point in mutating if there is only one activation function

        eligible_nodes = list(self.hidden_nodes().values())
        if config.output_activation is None:
            eligible_nodes.extend(self.output_nodes().values())
        if config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes().values())
        for node in eligible_nodes:
            if torch.rand(1)[0] < prob:
                node.set_activation(random_choice(config.activations))
        self.outputs = None # reset the image


    def mutate_weights(self, prob, config):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """
        R_delta = torch.rand(len(self.connection_genome.items()), device=self.device)
        R_reset = torch.rand(len(self.connection_genome.items()), device=self.device)

        for i, connection in enumerate(self.connection_genome.values()):
            if R_delta[i] < prob:
                delta = random_normal(None, 0, config.weight_mutation_std)
                connection.weight = connection.weight + delta
            elif R_reset[i] < config.prob_weight_reinit:
                connection.weight = self.random_weight()

        # self.clamp_weights()
        self.outputs = None # reset the image


    def mutate_bias(self, prob, config):
        R_delta = torch.rand(len(self.node_genome.items()), device=self.device)
        R_reset = torch.rand(len(self.node_genome.items()), device=self.device)

        for i, node in enumerate(self.node_genome.values()):
            if R_delta[i] < prob:
                delta = random_normal(None, 0, config.bias_mutation_std)
                node.bias = node.bias + delta
            elif R_reset[i] < config.prob_weight_reinit:
                node.bias = torch.zeros_like(node.bias)

        self.outputs = None # reset the image
        
        
    def mutate_lr(self, sigma):
        if not sigma:
            return # don't mutate
        self.sgd_lr = self.sgd_lr + random_normal(None, 0, sigma)
        self.sgd_lr = max(1e-8, self.sgd_lr)
        self.outputs = None
   

    def mutate(self, config, rates=None):
        """Mutates the CPPN based on its config or the optionally provided rates."""
        if rates is None:
            add_node = config.prob_add_node
            add_connection = config.prob_add_connection
            remove_node = config.prob_remove_node
            disable_connection = config.prob_disable_connection
            mutate_weights = config.prob_mutate_weight
            mutate_bias = config.prob_mutate_bias
            mutate_activations = config.prob_mutate_activation
            mutate_sgd_lr_sigma = config.mutate_sgd_lr_sigma
        else:
            mutate_activations, mutate_weights, mutate_bias, add_connection, add_node, remove_node, disable_connection, weight_mutation_max, prob_reenable_connection = rates
        
        rng = lambda: random_uniform(None,0.0,1.0)
        for _ in range(config.mutation_iters):
            if config.single_structural_mutation:
                div = max(1.0, (add_node + remove_node +
                                add_connection + disable_connection))
                r = rng()
                if r < (add_node / div):
                    self.add_node(config)
                elif r < ((add_node + remove_node) / div):
                    self.remove_node(config)
                elif r < ((add_node + remove_node +
                            add_connection) / div):
                    self.add_connection(config)
                elif r < ((add_node + remove_node +
                            add_connection + disable_connection) / div):
                    self.disable_connection()
            else:
                # mutate each structural category separately
                if rng() < add_node:
                    self.add_node(config)
                if rng() < remove_node:
                    self.remove_node(config)
                if rng() < add_connection:
                    self.add_connection(config)
                if rng() < disable_connection:
                    self.disable_connection()
            
            self.mutate_activations(mutate_activations, config)
            self.mutate_weights(mutate_weights, config)
            self.mutate_bias(mutate_bias, config)
            self.mutate_lr(mutate_sgd_lr_sigma)
            self.update_node_layers()
            self.disable_invalid_connections(config)
            
        self.graph = {}
            
            
        self.outputs = None # reset the image
        if hasattr(self, 'aot_fn'):
            del self.aot_fn # needs recompile

    def disable_invalid_connections(self, config):
        """Disables connections that are not compatible with the current configuration."""
        # return # TODO: test, but there should never be invalid connections
        invalid = []
        for key, connection in self.connection_genome.items():
            if connection.enabled:
                if not is_valid_connection(self.node_genome,
                                           [k.split(',') for k in self.connection_genome.keys()],
                                           key.split(','),
                                           config,
                                           warn=True):
                    invalid.append(key)
        for key in invalid:
            del self.connection_genome[key]

    def add_connection(self, config):
        """Adds a connection to the CPPN."""
        self.update_node_layers()
        
        for _ in range(20):  # try 20 times max
            [from_node, to_node] = random_choice(list(self.node_genome.values()),
                                                 2, replace=False)
            if from_node.layer >= to_node.layer:
                continue  # don't allow recurrent connections
            # look to see if this connection already exists
            key = f"{from_node.id},{to_node.id}"
            if key in self.connection_genome.keys():
                existing_cx = self.connection_genome[key]
            else:
                existing_cx = None
            
            # if it does exist and it is disabled, there is a chance to reenable
            if existing_cx is not None:
                if not existing_cx.enabled:
                    if torch.rand(1)[0] < config.prob_reenable_connection:
                        existing_cx.enabled = True # re-enable the connection
                    break  # don't enable more than one connection
                continue # don't add more than one connection

            # else if it doesn't exist, check if it is valid
            # if is_valid_connection(self.node_genome,  self.connection_genome, (from_node.id, to_node.id), config):
            if is_valid_connection(self.node_genome,
                                           [k.split(',') for k in self.connection_genome.keys()],
                                           key.split(','),
                                           config):
                # valid connection, add
                new_cx = Connection(key, self.random_weight(False, config.weight_init_std))
                assert new_cx.key not in self.connection_genome.keys(),\
                    "CX already exists: {}".format(new_cx.key)
                self.connection_genome[new_cx.key] = new_cx
                self.update_node_layers()
                break # found a valid connection, don't add more than one

            # else failed to find a valid connection, don't add and try again
        self.outputs = None # reset the image

    def add_node(self, config):
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
        old_connection = random_choice(eligible_cxs, 1, replace=False)

        # create the new node
        new_node = Node(self.get_new_node_id(), random_choice(config.activations),
                        NodeType.HIDDEN, 999, config.node_agg, device=self.device, grad=False)
        
        assert new_node.id not in self.node_genome.keys(),\
            "Node ID already exists: {}".format(new_node.id)
            
        self.node_genome[new_node.id] =  new_node # add a new node between two nodes

        old_connection.enabled = False  # disable old connection

        # The connection between the first node in the chain and the
        # new node is given a weight of one and the connection between
        # the new node and the last node in the chain
        # is given the same weight as the connection being split
        old_cx_key = old_connection.key_tuple
        new_cx_1 = Connection(
            f"{old_cx_key[0]},{new_node.id}", torch.tensor(1.0, device=self.device, dtype=dtype, requires_grad=False))
        assert new_cx_1.key not in self.connection_genome.keys()
        self.connection_genome[new_cx_1.key] = new_cx_1

        new_cx_2 = Connection(f"{new_node.id},{old_cx_key[1]}",
            old_connection.weight)
        assert new_cx_2.key not in self.connection_genome.keys()
        self.connection_genome[new_cx_2.key] = new_cx_2

        self.update_node_layers() # update the layers of the nodes
        self.outputs = None # reset the image
        
    def remove_node(self, config):
        """Removes a node from the CPPN.
            Only hidden nodes are eligible to be removed.
        """

        hidden = self.hidden_nodes().values()
        if len(hidden) == 0:
            return # no eligible nodes, don't remove a node

        # choose a random node
        node_id_to_remove = random_choice([n.id for n in hidden], 1, False)

        for key, cx in list(self.connection_genome.items())[::-1]:
            if node_id_to_remove in cx.key_tuple:
                del self.connection_genome[key]
        for key, node in list(self.node_genome.items())[::-1]:
            if node.id == node_id_to_remove:
                del self.node_genome[key]
                break

        self.update_node_layers()
        self.disable_invalid_connections(config)
        self.outputs = None # reset the image

    def prune(self, config):
        removed = 0
        for cx in list(self.connection_genome.values())[::-1]:
            if abs(cx.weight)< config.prune_threshold:
                del self.connection_genome[cx.key]
                removed += 1
        for _ in range(config.min_pruned - removed):
            min_weight_key = min(self.connection_genome, key=lambda k: self.connection_genome[k].weight.item())
            removed += 1
            del self.connection_genome[min_weight_key]
       
        # remove nodes with no connections
        all_keys = []
        all_keys.extend([cx.key.split(',')[0] for cx in self.connection_genome.values()])
        all_keys.extend([cx.key.split(',')[1] for cx in self.connection_genome.values()])

        for node in list(self.node_genome.values())[::-1]:
            if node.id not in all_keys:
                del self.node_genome[node.id]
        
        # print("Pruned {} connections".format(removed))
        
        self.update_node_layers()
        self.disable_invalid_connections(config)
        self.outputs = None # reset the image

    
    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections())
        if len(eligible_cxs) < 1:
            return
        cx = random_choice(eligible_cxs, 1, False)
        cx.enabled = False
        self.outputs = None # reset the image

    def update_node_layers(self):
        """Update the node layers."""
        layers = feed_forward_layers(self)
        max_layer = 0
        for _, node in self.input_nodes().items():
            node.layer = 0
        for layer_index, layer in enumerate(layers):
            for node_id in layer:
                node = find_node_with_id(self.node_genome, node_id)
                node.layer = layer_index + 1
            max_layer = max(max_layer, layer_index + 1)
            
        self.graph = {}
         

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

    def clamp_weights(self, config):
        """Clamps all weights to the range [-max_weight, max_weight]."""
        if not config.get("clamp_weights", True):
            return
            
        for _, cx in self.connection_genome.items():
            if cx.weight < config.weight_threshold and cx.weight >\
                 -config.weight_threshold:
                cx.weight = torch.tensor(0.0, device=self.device, requires_grad=cx.weight.requires_grad)
            if not isinstance(cx.weight, torch.Tensor):
                cx.weight = torch.tensor(cx.weight, device=self.device, requires_grad=cx.weight.requires_grad)
            cx.weight = torch.clamp(cx.weight, min=-config.max_weight, max=config.max_weight)
            
    def clear_data(self):
        """Clears the data from the network."""
        self.outputs = None
        for node in self.node_genome.values():
            node.sum_inputs = None
            node.outputs = None
        
        for connection in self.connection_genome.values():
            connection.weight.grad = None
    
    def reset_activations(self, shape):
        """Resets all node activations to zero."""
        for _, node in self.node_genome.items():
            node.sum_inputs = torch.zeros(shape, device=self.device)
            node.outputs = torch.zeros(shape, device=self.device)

 
    def forward(self, inputs=None, channel_first=True, act_mode='node', use_graph=False):
        res_h, res_w = inputs.shape[0], inputs.shape[1]
        node_shape = (res_h, res_w)
        
        if str(self.device)!=str(inputs.device):
            logging.warning(f"Moving CPPN to inputs device: {inputs.device}")
            self.to(inputs.device) # breaks computation graph
            
        # reset the activations to 0 before evaluating
        self.reset_activations(node_shape)
        
        # get layers
        layers = feed_forward_layers(self) 
        layers.insert(0, self.input_nodes().keys()) # add input nodes as first layer

        # iterate over layers
        for layer in layers:
            Xs, Ws, nodes= [], [], []
            for node_index, node_id in enumerate(layer):
                # iterate over nodes in layer
                node = self.node_genome[node_id] # the current node
                
                if node.type == NodeType.INPUT:
                    # initialize the node's sum
                    X = inputs[:,:,node_index].unsqueeze(-1)
                    weights = torch.ones((1), dtype=dtype, device=self.device)
                else:
                    # find incoming connections and activate
                    if use_graph and node_id in self.graph.keys():
                        required_cxs = self.graph[node_id] # use cached
                        X, weights = cx_ids_to_inputs(required_cxs, self.node_genome)
                    else:
                        required_cxs = collect_connections(self, node_id)
                        self.graph[node_id] = required_cxs 
                        
                    X, weights = cx_ids_to_inputs(required_cxs, self.node_genome)
                        
                    # X shape = (num_incoming, res_h, res_w)
                    if X is None:
                        X = torch.zeros((res_h, res_w), dtype=dtype, device=self.device)
                    if weights is None:
                        weights = torch.ones((1), dtype=dtype, device=self.device)

                if act_mode == 'node':
                    assert torch.isfinite(X).all()
                    if not torch.isfinite(weights).all():
                        # self.draw_nx(show=True)
                        logging.warning(f"found {torch.tensor(torch.isfinite(weights)==0.0).sum()} non-finite values in node {node.id} weights (out of {weights.numel()}), {weights}")
                    assert torch.isfinite(weights).all(), f"found {torch.tensor(torch.isfinite(weights)==0.0).sum()} non-finite values in node {node.id} weights (out of {weights.numel()}), {weights}"
                    # weights[~torch.isfinite(weights)] = torch.nn.Parameter(torch.tensor(0.0, device=self.device, dtype=dtype))
                    
                    node.activate(X, weights) # naive
                    
                    assert torch.isfinite(node.outputs).all(), f"Node {node.id} with activation {node.activation} and inputs {X}, weights {weights} has non-finite output {node.outputs}"
                elif act_mode == 'layer':
                    # group by function for efficiency
                    Xs.append(X)
                    Ws.append(weights)
                    nodes.append(node)
                elif act_mode == 'population':
                    # group by function for efficiency
                    raise RuntimeError("individual forward() called for population activation mode.")
                else:
                    raise ValueError(f"Unknown activation mode {act_mode}")
                
            if act_mode == 'layer':
                node_agg = nodes[0].agg
                activate_layer(Xs, Ws, {n.id:n for n in nodes}, node_agg)
            
        # collect outputs from the last layer
        sorted_o = sorted(self.output_nodes().values(), key=lambda x: x.key, reverse=True)
        outputs = torch.stack([node.outputs for node in sorted_o], dim=0 if channel_first else -1)
        assert str(outputs.device) == str(self.device), f"Output is on {outputs.device}, should be {self.device}"

        self.outputs = outputs
                
        if self.output_blur > 0:
            self.outputs = gaussian_blur(self.outputs, self.output_blur)
                
        assert str(self.outputs.device )== str(self.device), f"Output is on {self.outputs.device}, should be {self.device}"
        assert self.outputs.dtype == torch.float32, f"Output is {self.outputs.dtype}, should be float32"
    
        return self.outputs
    

    def backward(self, loss:torch.Tensor,retain_graph=False):
        """Backpropagates the error through the network."""
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        self.outputs = None # new image
    
    
    def discard_grads(self):
        return
        for _, cx in self.connection_genome.items():
            # check nan
            if isinstance(cx.weight, float):
                cx.weight = torch.tensor(cx.weight, device=self.device, dtype=dtype)
            if torch.isnan(cx.weight).any():
                # TODO: why NaN?
                cx.weight = torch.tensor(0, device=self.device)
            else:
                cx.weight = torch.tensor(cx.weight.detach().item(), device=self.device, dtype=dtype)
        for _, node in self.node_genome.items():
            if isinstance(node.bias, float):
                node.bias = torch.tensor(node.bias, device=self.device, dtype=dtype)
            node.bias = torch.tensor(node.bias.detach().item(), device=self.device, dtype=dtype)
        # self.reset_activations()
        for _, node in self.node_genome.items():
            node.sum_inputs = None
            node.outputs = None
        
        self.outputs = None # new image

        if hasattr(self, 'optimizer'):
            del self.optimizer
            self.optimizer = None


    def crossover(self, other):
        """ Configure a new genome by crossover from two parent genomes. """
        
        # TODO: this may mess up the order of the outputs, need to check
        
        child = type(self)(self.config, {}, {}) # create an empty child genome
        assert self.fitness is not None, "Parent 1 has no fitness"
        assert other.fitness is not None, "Parent 2 has no fitness"
        # determine which parent is more fit
        if self.fitness > other.fitness:
            parent1, parent2 = self, other
        elif self.fitness < other.fitness:
            parent1, parent2 = other, self
        else:
            # fitness same, choose randomly
            if torch.rand(1)[0] < 0.5:
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


    # def __copy__(self):
        # return self.clone()


    def clone(self, config, cpu=False, new_id=False):
        """ Create a copy of this genome. """
        
        id = self.id if (not new_id) else type(self).get_id()

        child = type(self)(config, nn.ModuleDict(), nn.ModuleDict())
        
        import io
        buffer = io.BytesIO()
        torch.save(self, buffer) #<--- model is some nn.module

        # read from buffer
        buffer.seek(0) #<--- must see to origin every time before reading
        child = torch.load(buffer)
        del buffer
        
        
        
        # child.connection_genome = nn.ModuleDict({key: copy.copy(cx) for key, cx in self.connection_genome.items()})
        # child.node_genome = nn.ModuleDict({key: copy.copy(node) for key, node in self.node_genome.items()})
        # # load_state_dict:
        # child.load_state_dict(self.state_dict())
        
        # for n in child.node_genome.values():
        #     n.bias = torch.tensor(n.bias.item(), dtype=dtype, device=child.device)
        
        # child.update_node_layers()
        # for cx in child.connection_genome.values():
        #     cx.weight = torch.tensor(cx.weight.item(), dtype=dtype, device=self.device) # require prepare_optimizer() again
        
        if new_id:
            child.parents = (self.id, self.id)
            child.lineage = self.lineage + [self.id]
        else:
            child.parents = self.parents
            child.lineage = self.lineage
            
        
        child.set_id(id)
        child.age = 0
        child.graph = {}
        
        if cpu:
            child.to('cpu')
        else:
            child.to(self.device)

        
        return child
    
    # def to(self, device):
    #     """Moves the CPPN to the given device (in-place)."""
    #     if isinstance(device, str):
    #         device = torch.device(device)
    #     # if self.device == device:
    #         # return
    #     self.device = device
    #     for key, node in self.node_genome.items():
    #         node.to(device)
    #     for key, cx in self.connection_genome.items():
    #         cx.to(device)
                

    def __call__(self, *args, **kwargs):
        # wrapper for forward
        return self.forward(*args, **kwargs)
    

if __name__== "__main__":
    from config import CPPNConfig
    c = CPPNConfig()
    c.device = torch.device('cuda:0')
    c.color_mode = 'L'
    cppn = CPPN(c)
    for _ in range(100):
        cppn.mutate(c)
    inputs = CPPN.initialize_inputs_from_config(c)
    outputs = cppn(inputs, channel_first=False)
    
    from util import visualize_network
    
    import matplotlib.pyplot as plt
    visualize_network(cppn, c)
    
    
    
    plt.imshow(outputs.detach().cpu().numpy(), cmap='gray')
    plt.show()


    child = cppn.clone(c)
    visualize_network(child, c)
    
    outputs = child(inputs, channel_first=False)
    plt.imshow(outputs.detach().cpu().numpy(), cmap='gray')
    plt.show()        
    