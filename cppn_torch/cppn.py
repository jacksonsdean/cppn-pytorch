"""Contains the CPPN, Node, and Connection classes."""
import copy
from enum import IntEnum
from itertools import count
import math
import json
import os
from typing import Callable, List
from typing import Union
import torch

from functorch.compile import compiled_function, draw_graph, aot_function
from cppn_torch.activation_functions import identity
from cppn_torch.graph_util import *
from cppn_torch.config import CPPNConfig as Config
from cppn_torch.gene import * 

def random_uniform(low=0.0, high=1.0, device: Union[str,torch.device] =torch.device('cpu'), grad=False):
    return torch.rand(1, device=device, requires_grad=grad)[0] * (high - low) + low

def random_choice(choices, count, replace):
    if not replace:
        indxs = torch.randperm(len(choices))[:count]
        output = []
        for i in indxs:
            output.append(choices[i])
        return output
    else:
        return [random.choice(choices) for _ in range(count)]

class CPPN():
    """A CPPN Object with Nodes and Connections."""

    constant_inputs = torch.zeros((0, 0, 0), dtype=torch.float32,requires_grad=False) # (res_h, res_w, n_inputs)
    current_id = 1 # 0 reserved for 'random' parent
    node_indexer = None
    
    
    @staticmethod
    def initialize_inputs(res_h, res_w, use_radial_dist, use_bias, n_inputs, device, coord_range=(-.5,.5) ):
        """Initializes the pixel inputs."""

        # Pixel coordinates are linear within coord_range
        x_vals = torch.linspace(coord_range[0], coord_range[1], res_w, device=device,dtype=torch.float32)
        y_vals = torch.linspace(coord_range[0], coord_range[1], res_h, device=device,dtype=torch.float32)

        # initialize to 0s
        CPPN.constant_inputs = torch.zeros((res_h, res_w, n_inputs), dtype=torch.float32, device=device, requires_grad=False)

        # assign values:
        for y in range(res_h):
            for x in range(res_w):
                this_pixel = [y_vals[y], x_vals[x]] # coordinates
                if use_radial_dist:
                    # d = sqrt(x^2 + y^2)
                    this_pixel.append(torch.tensor(math.sqrt(y_vals[y]**2 + x_vals[x]**2)))
                if use_bias:
                    this_pixel.append(torch.tensor(1.0)) # bias = 1.0
                CPPN.constant_inputs[y][x] = torch.tensor(this_pixel, dtype=torch.float32, device=device, requires_grad=False)

    def __init__(self, config = None, nodes = None, connections = None) -> None:
        self.config = config
        if self.config is None:
            self.config = Config()
            
        self.outputs = None
        self.node_genome = {}
        self.connection_genome = {}
        self.selected = False
        self.species_id = 0
        self.id = CPPN.get_id()
        
        self.reconfig(self.config, nodes, connections)
        
        self.parents = (0, 0)
        self.fitness = torch.tensor(0.0, device=self.device)
        self.novelty = torch.tensor(0.0, device=self.device)
        self.adjusted_fitness = torch.tensor(0.0, device=self.device)
    
    def reconfig(self, config = None, nodes = None, connections = None):
        if config is not None:
            self.config = config
        assert self.config is not None
     
        self.device = self.config.device
        torch.manual_seed(self.config.seed)
        if self.device is None:
            raise ValueError("device is None") # TODO
            # no device specified, try to use GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.n_inputs = 2 # x, y
        if self.config.use_radial_distance:
            self.n_inputs += 1 # x, y, d
        if self.config.use_input_bias:
            self.n_inputs+=1 # x, y, d?, bias

        self.config.num_inputs = self.n_inputs # TODO not sure
        self.n_outputs = len(self.config.color_mode) # RGB (3), HSV(3), L(1)

        if nodes is None:
            self.initialize_node_genome()
        else:
            self.node_genome = nodes
        if connections is None:
            self.initialize_connection_genome()
        else:
            self.connection_genome = connections
            
        self.config._not_dirty()
    
    @staticmethod
    def get_id():
        CPPN.current_id += 1
        return CPPN.current_id - 1

    def get_new_node_id(self):
        """Returns a new node id`."""
        if CPPN.node_indexer is None:
            if self.node_genome == {}:
                return 0
            if self.node_genome is not None:
                CPPN.node_indexer = count(max(list(self.node_genome)) + 1)
            else:
                CPPN.node_indexer = count(max(list(self.node_genome)) + 1)

        new_id = next(CPPN.node_indexer)
        assert new_id not in self.node_genome.keys()
        return new_id
    
    def set_id(self, id):
        self.id = id
        
    def initialize_node_genome(self):
        """Initializes the node genome."""
        assert self.config is not None
        total_node_count = self.n_inputs + self.config.num_extra_inputs +\
            self.n_outputs + self.config.hidden_nodes_at_start
        for idx in range(self.n_inputs + self.config.num_extra_inputs):
            new_node = Node(-(1+idx), identity, NodeType.INPUT, 0)
            self.node_genome[new_node.key] = new_node
            
        for idx in range(self.n_inputs, self.n_inputs + self.n_outputs):
            if self.config.output_activation is None:
                output_fn = choose_random_function(self.config)
            else:
                output_fn = self.config.output_activation
            new_node = Node(-(1+idx), output_fn, NodeType.OUTPUT, 2)
            self.node_genome[new_node.key] = new_node
            
        for _ in range(self.n_inputs + self.n_outputs, total_node_count):
            new_node = Node(self.get_new_node_id(), choose_random_function(self.config),
                            NodeType.HIDDEN, 1)
            self.node_genome[new_node.key] = new_node
     
     
    def get_output_layer_index(self):
        for n in self.node_genome.values():
            if n.type == NodeType.OUTPUT:
                return n.layer
        assert False, "No output layer found"
    
    def get_params(self):
        """Returns a list of all parameters in the network."""
        params = []
        for cx in self.connection_genome.values():
            params.append(cx.weight)
        return params

    def get_named_params(self):
        """Returns a dict of all parameters in the network."""
        params = {}
        for cx in self.connection_genome.values():
            params[f"w_{cx.key}"] = cx.weight
        return params

    def prepare_optimizer(self, opt_class=torch.optim.Adam, lr=None):
        """Prepares the optimizer."""
        assert self.config is not None, "Config is None."
        if lr is None:
            lr = self.config.sgd_learning_rate
        self.outputs = None
        for cx in self.connection_genome.values():
            cx.weight = cx.weight.detach()
            cx.weight = torch.nn.Parameter(torch.tensor(cx.weight.item(), requires_grad=True))
        self.optimizer = opt_class([cx.weight for cx in self.connection_genome.values()], lr=lr)

    

    def initialize_connection_genome(self):
        """Initializes the connection genome."""
        assert self.config is not None, "Config is None."

        output_layer_idx = self.get_output_layer_index()
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
        
    def serialize(self):
        del CPPN.constant_inputs
        assert self.config is not None, "Config is None."
        CPPN.constant_inputs = None
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
        CPPN.initialize_inputs(self.config.res_h, self.config.res_w,
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
        # return random_uniform(-self.config.max_weight, self.config.max_weight, self.device, grad=True)
        return random_uniform(-self.config.max_weight, self.config.max_weight, self.device, grad=False)

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

        eligible_nodes = list(self.hidden_nodes().values())
        if self.config.output_activation is None:
            eligible_nodes.extend(self.output_nodes().values())
        if self.config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes().values())
        for node in eligible_nodes:
            if random_uniform(0,1) < prob:
                node.activation = choose_random_function(self.config)
        self.outputs = None # reset the image

    def mutate_weights(self, prob):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """
        assert self.config is not None, "Config is None."

        for _, connection in self.connection_genome.items():
            if random_uniform(0, 1) < prob:
                delta = random_uniform(-self.config.weight_mutation_max,
                                               self.config.weight_mutation_max, device=self.device)
                connection.weight = connection.weight + delta
            elif random_uniform(0, 1) < self.config.prob_weight_reinit:
                connection.weight = self.random_weight()

        self.clamp_weights()
        self.outputs = None # reset the image
        

    def mutate(self, rates=None):
        """Mutates the CPPN based on its config or the optionally provided rates."""
        self.fitness, self.adjusted_fitness, self.novelty = torch.tensor(0.0,device=self.device), torch.tensor(0.0,device=self.device), torch.tensor(0.0,device=self.device) # new fitnesses after mutation
        assert self.config is not None, "Config is None."
        if rates is None:
            add_node = self.config.prob_add_node
            add_connection = self.config.prob_add_connection
            remove_node = self.config.prob_remove_node
            disable_connection = self.config.prob_disable_connection
            mutate_weights = self.config.prob_mutate_weight
            mutate_activations = self.config.prob_mutate_activation
        else:
            mutate_activations, mutate_weights, add_connection, add_node, remove_node, disable_connection, weight_mutation_max, prob_reenable_connection = rates
        if random_uniform(0.0,1.0) < add_node:
            self.add_node()
        if random_uniform(0.0,1.0) < remove_node:
            self.remove_node()
        if random_uniform(0.0,1.0) < add_connection:
            self.add_connection()
        if random_uniform(0.0,1.0) < disable_connection:
            self.disable_connection()

        self.mutate_activations(mutate_activations)
        self.mutate_weights(mutate_weights)
        self.update_node_layers()
        # self.disable_invalid_connections()
        self.outputs = None # reset the image
        if hasattr(self, 'aot_fn'):
            del self.aot_fn # needs recompile
        

    def disable_invalid_connections(self):
        """Disables connections that are not compatible with the current configuration."""
        return # TODO?
        for key, connection in self.connection_genome.items():
            if connection.enabled:
                if not is_valid_connection(self.node_genome, connection.key, self.config):
                    connection.enabled = False

    def add_connection(self):
        """Adds a connection to the CPPN."""
        assert self.config is not None, "Config is None."
        
        for _ in range(20):  # try 20 times max
            [from_node, to_node] = random_choice(
                list(self.node_genome.values()), 2, replace=False)

            # look to see if this connection already exists
            existing_cx = self.connection_genome.get((from_node.id, to_node.id))

            # if it does exist and it is disabled, there is a chance to reenable
            if existing_cx is not None:
                if not existing_cx.enabled:
                    if random_uniform() < self.config.prob_reenable_connection:
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
        old_connection = random_choice(eligible_cxs,1,replace=False)[0]

        # create the new node
        new_node = Node(self.get_new_node_id(), choose_random_function(self.config),
                        NodeType.HIDDEN, 999)
        
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
            (old_connection.key[0], new_node.id), torch.tensor(1.0, device=self.device,dtype=torch.float32))
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
        node_id_to_remove = random_choice([n.id for n in hidden], 1, False)[0]

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

    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections())
        if len(eligible_cxs) < 1:
            return
        cx = random_choice(eligible_cxs, 1, False)[0]
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
        
        for _, cx in self.connection_genome.items():
            if cx.weight < self.config.weight_threshold and cx.weight >\
                 -self.config.weight_threshold:
                cx.weight = torch.tensor(0.0, device=self.device, requires_grad=cx.weight.requires_grad)
            if not isinstance(cx.weight, torch.Tensor):
                cx.weight = torch.tensor(cx.weight, device=self.device, requires_grad=cx.weight.requires_grad)
            cx.weight = torch.clamp(cx.weight, -self.config.max_weight, self.config.max_weight)
            

    def reset_activations(self, parallel=True):
        """Resets all node activations to zero."""
        assert self.config is not None, "Config is None."

        if parallel:
            for _, node in self.node_genome.items():
                node.sum_inputs = torch.ones((1, self.config.res_h, self.config.res_w), device=self.device)/2.0
                node.outputs = torch.ones((1, self.config.res_h, self.config.res_w), device=self.device)/2.0
        else:
            for _, node in self.node_genome.items():
                node.sum_inputs = torch.zeros(1, device=self.device)
                node.outputs = torch.zeros(1, device=self.device)
                
    def reset_grads(self):
        for _, cx in self.connection_genome.items():
            if cx.weight.grad is not None:
                cx.weight.grad.zero_()
    
            
    def eval(self, inputs, extra_inputs=None):
        """Evaluates the CPPN."""
        self.set_inputs(inputs)
        return self.forward_(extra_inputs)

    def forward_(self, extra_inputs=None):
        assert self.config is not None, "Config is None."
        
        batch_size = 1 if extra_inputs is None else extra_inputs.shape[0]
        res_h, res_w = self.config.res_h, self.config.res_w
        
        if extra_inputs is not None:
            if len(extra_inputs.shape) == 1:
                raise RuntimeError("Extra inputs must be (batch_size, num_extra_inputs)")
            if len(extra_inputs.shape) == 2:
                # repeat for the whole image
                # we want (batch_size, res_h, res_w, num_extra_inputs)
                extra_inputs = extra_inputs.unsqueeze(1).unsqueeze(2).repeat(1, res_h, res_w, 1)
            
        # reset the activations to 0 before evaluating
        self.reset_activations()
        
        # get layers
        layers = feed_forward_layers(self) 
        layers.insert(0, self.input_nodes().keys()) # add input nodes as first layer
        
        starting_input = None
        for layer in layers:
            # iterate over layers
            for node_index, node_id in enumerate(layer):
                # iterate over nodes in layer
                node = find_node_with_id(self.node_genome, node_id) # the current node

                # find incoming connections
                node_inputs = get_incoming_connections(self, node)

                # initialize the node's sum_inputs
                if node.type == NodeType.INPUT:
                    if node_index < self.config.num_inputs:
                        starting_input = CPPN.constant_inputs[:,:,node_index].repeat(batch_size, 1, 1) # (batch_size, res_h, res_w)
                    elif extra_inputs is not None:
                        # we want (batch_size, res_h, res_w)
                        this_idx = node_index - self.config.num_inputs
                        starting_input = extra_inputs[:,:,:,this_idx]
                else:
                    # not an input node
                    starting_input = torch.zeros((batch_size, res_h, res_w), dtype=torch.float32, device=self.device)

                node.initialize_sum(starting_input)
                # initialize the sum_inputs for this node
                node.activate(node_inputs, self.node_genome)

        # collect outputs from the last layer
        outputs = torch.stack([node.outputs for node in self.output_nodes().values()])
        assert str(outputs.device) == str(self.device), f"Output is on {outputs.device}, should be {self.device}"

        self.outputs = outputs
        
        assert str(self.outputs.device )== str(self.device), f"Output is on {self.outputs.device}, should be {self.device}"
        assert self.outputs.dtype == torch.float32, f"Output is {self.outputs.dtype}, should be float32"
        return self.outputs
    
    def forward(self, extra_inputs=None):
        """Feeds forward the network."""
        
        assert self.config is not None, "Config is None."
        if self.config.with_grad:
            # self.reset_grads()
            
            if not hasattr(self, 'aot_fn'):
                def f(x):
                    return self.forward_(extra_inputs=x) 
                def fw(f, inps):
                    return f
                def bw(f, inps):
                    return f
                
                self.aot_fn = aot_function(f, fw_compiler=fw, bw_compiler=bw)
            
            return self.aot_fn(extra_inputs)  
        else:  
            return self.forward_(extra_inputs=extra_inputs)

    def backward(self, loss:torch.Tensor,retain_graph=False):
        """Backpropagates the error through the network."""
        assert self.config is not None
        assert self.config.with_grad, "Cannot backpropagate without gradients."
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        self.outputs = None # new image
    
    def discard_grads(self):
        for _, cx in self.connection_genome.items():
            if cx.weight.grad is not None:
                # cx.weight.grad.zero_()
                cx.weight = cx.weight.detach()
                # cx.weight.requires_grad = False
        self.outputs = None # new image
        self.fitness= self.fitness.detach()
        self.optimizer = None
        
    
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
            abs(o_cx.weight-t_cx.weight) for o_cx, t_cx in zip(other_matching, this_matching)]
        # difference_of_matching_weights = torch.stack(difference_of_matching_weights)
        
        if(len(difference_of_matching_weights) == 0):
            difference_of_matching_weights = 0
        else:
            difference_of_matching_weights = torch.mean(
                torch.stack(difference_of_matching_weights)).item()

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
        difference = n_excess + n_disjoint + \
            difference_of_matching_weights + n_different_fns

        return difference

    def species_comparision(self, other, threshold) -> bool:
        # returns whether other is the same species as self
        return self.genetic_difference(other) < threshold


    def crossover(self, other):
        """ Configure a new genome by crossover from two parent genomes. """
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
            
    def to_cpu(self):
        '''Move all tensors to CPU'''
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

    @torch.no_grad()
    def clone(self, deepcopy=True, cpu=False, new_id=False):
        """ Create a copy of this genome. """
        id = self.id if (not new_id) else CPPN.get_id()
        if deepcopy:
            child = copy.deepcopy(self)
            child.set_id(id)
            if cpu:
                child.to_cpu()
            return child
        child = CPPN(self.config, {}, {})
        child.connection_genome = {key: cx.copy() for key, cx in self.connection_genome.items()}        
        child.node_genome = {key: node.copy() for key, node in self.node_genome.items()}
        child.update_node_layers()
        for cx in child.connection_genome.values():
            # detach from current graph
            has_grad = cx.weight.requires_grad
            cx.weight = cx.weight.detach()
            cx.weight = torch.tensor(cx.weight.item(), requires_grad=has_grad)
        if cpu:
            child.to_cpu()
        child.set_id(id)
        return child

   