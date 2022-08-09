"""Contains the CPPN, Node, and Connection classes."""
import copy
from enum import IntEnum
from itertools import count
import math
import json
from typing import Callable
# import numpy as np
import torch

from cppn_neat.util import visualize_network
try:
    from activation_functions import identity
    from graph_util import name_to_fn, choose_random_function, is_valid_connection
    from graph_util import *
    from graph_util import get_incoming_connections, feed_forward_layers
    from graph_util import hsv2rgb
except ModuleNotFoundError:
    from cppn_neat.activation_functions import identity
    from cppn_neat.graph_util import *
    from cppn_neat.graph_util import name_to_fn, choose_random_function, is_valid_connection
    from cppn_neat.graph_util import get_incoming_connections, feed_forward_layers
    from cppn_neat.graph_util import hsv2rgb


def random_uniform(low=0.0, high=1.0, device='cpu'):
    return torch.rand(1, device=device)[0] * (high - low) + low

def random_choice(choices, count, replace):
    if not replace:
        indxs = torch.randperm(len(choices))[:count]
        output = []
        for i in indxs:
            output.append(choices[i])
        return output
    else:
        return choices[torch.randint(0, len(choices), (count,), device=choices.device)]

class NodeType(IntEnum):
    """Enum for the type of node."""
    INPUT  = 0
    OUTPUT = 1
    HIDDEN = 2

class Gene(object):
    """Represents either a node or connection in the CPPN"""
    def __init__(self) -> None:
        pass
    def copy(self):
        new_gene = self.__class__(self.key)
        for name, value in self._gene_attributes:
            setattr(new_gene, name, value)

        return new_gene

    @property
    def key(self):
        raise NotImplementedError
    @property
    def _gene_attributes(self) -> list:
        """A list of tuples of the form (name, value)"""
        raise NotImplementedError
    
    def crossover(self, other):
        assert self.key == other.key,\
            f"Cannot crossover genes with different keys {self.key!r} and {other.key!r}"
        new_gene = self.__class__(self.key)

        for name, value in self._gene_attributes:
            if torch.rand(1)[0] < 0.5:
                setattr(new_gene, name, value)
            else:
                setattr(new_gene, name, getattr(other, name))

        return new_gene
    
class Node(Gene):
    """Represents a node in the CPPN."""
    # TODO: aggregation function, bias, response(?)
    
    @staticmethod
    def create_from_json(json_dict):
        """Constructs a node from a json dict or string."""
        i = Node(None, None, None, None)
        i = i.from_json(json_dict)
        return i

    @staticmethod
    def empty():
        """Returns an empty node."""
        return Node(0, identity, NodeType.HIDDEN, 0)

    def __init__(self, key, activation=None, _type=2, _layer=999) -> None:
        self.activation = activation
        self.id = key
        self.type = _type
        self.layer = _layer
        self.sum_inputs = None
        self.outputs = None
        super().__init__()
    
    @property
    def key(self):
        return self.id
    @property
    def _gene_attributes(self):
        return [('activation', self.activation), ('type', self.type)]
    
    def activate(self, incoming_connections, nodes):
        """Activates the node given a list of connections that end here."""
        for cx in incoming_connections:
            if nodes[cx.key[0]] is not None:
                inputs = nodes[cx.key[0]].outputs * cx.weight
                self.sum_inputs = self.sum_inputs + inputs
        
        self.outputs = self.activation(self.sum_inputs)  # apply activation

    def initialize_sum(self, initial_sum):
        """Activates the node."""
        self.sum_inputs = initial_sum

    def serialize(self):
        """Makes the node serializable."""
        self.type = self.type.value if isinstance(self.type, NodeType) else self.type
        self.id = int(self.id)
        self.layer = int(self.layer)
        self.sum_inputs = None
        self.outputs = None
        try:
            self.activation = self.activation.__name__
        except AttributeError:
            pass
    
    def deserialize(self):
        """Makes the node functional"""
        self.sum_inputs = None
        self.sum_inputs = None
        self.activation = name_to_fn(self.activation) if isinstance(self.activation, str) else self.activation
        self.type = NodeType(self.type)

    def to_json(self):
        """Converts the node to a json string."""
        self.serialize()
        return json.dumps(self.__dict__)

    def from_json(self, json_dict):
        """Constructs a node from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        self.__dict__ = json_dict
        self.deserialize()
        assert isinstance(self.activation, Callable), "activation function is not a function"
        return self

class Connection(Gene):
    """
    Represents a connection between two nodes.

    where innovation number is the same for all of same connection
    i.e. 2->5 and 2->5 have same innovation number, regardless of individual
    """
    # innovations = []

    # @staticmethod
    # def get_innovation(key:tuple):
    #     """Returns the innovation number for the connection."""
    #     connection_from_to = key # based on id

    #     if connection_from_to in Connection.innovations:
    #         return Connection.innovations.index(connection_from_to)

    #     Connection.innovations.append(connection_from_to)
    #     return len(Connection.innovations) - 1

    # def __init__(self, from_node, to_node, weight, enabled = True) -> None:
    def __init__(self, key, weight = None, enabled = True) -> None:
        # Initialize
        self.key_ = key
        self.weight = weight
        # self.innovation = Connection.get_innovation(key)
        self.enabled = enabled
        # self.is_recurrent = to_node.layer < from_node.layer
        self.is_recurrent = False # TODO
        super().__init__()
        
    @property
    def key(self):
        assert type(self.key_) is tuple, "key is not a tuple"
        return self.key_
    @property
    def _gene_attributes(self):
        return [('weight', self.weight), ('enabled', self.enabled)]
    
    def serialize(self):
        self.weight = float(self.weight)
    def deserialize(self):
        pass
    
    def to_json(self):
        """Converts the connection to a json string."""
        # self.innovation = int(self.innovation)
        # if isinstance(self.from_node, Node):
            # self.from_node = self.from_node.to_json()
        # if isinstance(self.to_node, Node):
            # self.to_node = self.to_node.to_json()
        return json.dumps(self.__dict__)

    def from_json(self, json_dict):
        """Constructs a connection from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        self.__dict__ = json_dict
        # self.from_node = Node.create_from_json(self.from_node)
        # self.to_node = Node.create_from_json(self.to_node)
        return self

    @staticmethod
    def create_from_json(json_dict):
        """Constructs a connection from a json dict or string."""
        # f_node = Node.empty()
        # t_node = Node.empty()
        i = Connection((-1,-1), 0)
        i.from_json(json_dict)
        return i

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"([{self.key[0]}->{self.key[1]}]"+\
            f"W:{self.weight:3f} E:{self.enabled} R:{self.is_recurrent})"


class CPPN():
    """A CPPN Object with Nodes and Connections."""

    pixel_inputs = torch.zeros((0, 0, 0)) # (res_h, res_w, n_inputs)
    current_id = 0
    node_indexer = None
    
    
    @staticmethod
    def initialize_inputs(res_h, res_w, use_radial_dist, use_bias, n_inputs,device ):
        """Initializes the pixel inputs."""

        # Pixel coordinates are linear from -.5 to .5
        x_vals = torch.linspace(-.5, .5, res_w, device=device)
        y_vals = torch.linspace(-.5, .5, res_h, device=device)

        # initialize to 0s
        CPPN.pixel_inputs = torch.zeros((res_h, res_w, n_inputs), device=device)

        # assign values:
        for y in range(res_h):
            for x in range(res_w):
                this_pixel = [y_vals[y], x_vals[x]] # coordinates
                if use_radial_dist:
                    # d = sqrt(x^2 + y^2)
                    this_pixel.append(math.sqrt(y_vals[y]**2 + x_vals[x]**2))
                if use_bias:
                    this_pixel.append(1.0) # bias = 1.0
                CPPN.pixel_inputs[y][x] = torch.tensor(this_pixel, device=device)

    def __init__(self, config, nodes = None, connections = None) -> None:
        self.device = config.device
        if self.device is None:
            # no device specified, try to use GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.image = None
        self.node_genome = {}
        self.connection_genome = {}
        self.selected = False
        self.species_id = 0
        self.id = CPPN.get_id()
        
        torch.manual_seed(config.seed)

        self.n_inputs = 2 # x, y
        if config.use_radial_distance:
            self.n_inputs += 1 # x, y, d
        if config.use_input_bias:
            self.n_inputs+=1 # x, y, d?, bias

        self.n_outputs = len(config.color_mode) # RGB (3), HSV(3), L(1)

        if nodes is None:
            self.initialize_node_genome()
        else:
            self.node_genome = nodes
        if connections is None:
            self.initialize_connection_genome()
        else:
            self.connection_genome = connections
    
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
        total_node_count = self.n_inputs + \
            self.n_outputs + self.config.hidden_nodes_at_start
        for idx in range(self.n_inputs):
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
    
    def initialize_connection_genome(self):
        """Initializes the connection genome."""
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
        del CPPN.pixel_inputs
        CPPN.pixel_inputs = None
        if self.image is not None:
            self.image = self.image.cpu().numpy().tolist() if\
                isinstance(self.image, torch.Tensor) else self.image
        for _, node in self.node_genome.items():
            node.serialize()
        for _, connection in self.connection_genome.items():
            connection.serialize()
        self.config.serialize()

    def deserialize(self):
        for _, node in self.node_genome.items():
            node.deserialize()
        for _, connection in self.connection_genome.items():
            connection.deserialize()
        self.config.deserialize()
        
    def to_json(self):
        """Converts the CPPN to a json dict."""
        self.serialize()
        img = json.dumps(self.image) if self.image is not None else None
        # make copies to keep the CPPN intact
        copy_of_nodes = copy.deepcopy(self.node_genome).items()
        copy_of_connections = copy.deepcopy(self.connection_genome).items()
        return {"node_genome": [n.to_json() for _,n in copy_of_nodes], "connection_genome":\
            [c.to_json() for _,c in copy_of_connections], "image": img, "selected": self.selected}

    def from_json(self, json_dict):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        for key, value in json_dict.items():
            setattr(self, key, value)
       
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
        
        # make sure references are correct
        # for key, cx in self.connection_genome.items():
        #     cx.from_node = find_node_with_id(self.node_genome, cx.from_node.id)
        #     cx.to_node = find_node_with_id(self.node_genome, cx.to_node.id)

        self.update_node_layers()
        CPPN.initialize_inputs(self.config.res_h, self.config.res_w,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.n_inputs,
                self.device)

    @staticmethod
    def create_from_json(json_dict, config):
        """Constructs a CPPN from a json dict or string."""
        i = CPPN(config)
        i.from_json(json_dict)
        return i

    def random_weight(self):
        """Returns a random weight between -max_weight and max_weight."""
        return random_uniform(-self.config.max_weight, self.config.max_weight, self.device)

    def enabled_connections(self):
        """Returns a yield of enabled connections."""
        for key, connection in self.connection_genome.items():
            if connection.enabled:
                yield connection

    def mutate_activations(self, prob):
        """Mutates the activation functions of the nodes."""
        eligible_nodes = list(self.hidden_nodes().values())
        if self.config.output_activation is None:
            eligible_nodes.extend(self.output_nodes().values())
        if self.config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes().values())
        for node in eligible_nodes:
            if random_uniform(0,1) < prob:
                node.activation = choose_random_function(self.config)
        self.image = None # reset the image

    def mutate_weights(self, prob):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """

        for _, connection in self.connection_genome.items():
            if random_uniform(0, 1) < prob:
                connection.weight += random_uniform(-self.config.weight_mutation_max,
                                               self.config.weight_mutation_max, device=self.device)
            elif random_uniform(0, 1) < self.config.prob_weight_reinit:
                connection.weight = self.random_weight()

        self.clamp_weights()
        self.image = None # reset the image
        

    def mutate(self, rates=None):
        """Mutates the CPPN based on its config or the optionally provided rates."""
        self.fitness, self.adjusted_fitness = -torch.inf, -torch.inf # new fitnesses after mutation
        
        if rates is None:
            add_node = self.config.prob_add_node
            add_connection = self.config.prob_add_connection
            remove_node = self.config.prob_remove_node
            disable_connection = self.config.prob_disable_connection
            mutate_weights = self.config.prob_mutate_weight
            mutate_activations = self.config.prob_mutate_activation
        else:
            mutate_activations, mutate_weights, add_connection, add_node, remove_node, disable_connection, weight_mutation_max, prob_reenable_connection = rates
        if random_uniform(0,1) < add_node:
            self.add_node()
        if random_uniform(0,1) < remove_node:
            self.remove_node()
        if random_uniform(0,1) < add_connection:
            self.add_connection()
        if random_uniform(0,1) < disable_connection:
            self.disable_connection()

        self.mutate_activations(mutate_activations)
        self.mutate_weights(mutate_weights)
        self.update_node_layers()
        self.disable_invalid_connections()
        self.image = None # reset the image
        

    def disable_invalid_connections(self):
        """Disables connections that are not compatible with the current configuration."""
        return # TODO?
        for key, connection in self.connection_genome.items():
            if connection.enabled:
                if not is_valid_connection(self.node_genome, connection.key, self.config):
                    connection.enabled = False

    def add_connection(self):
        """Adds a connection to the CPPN."""
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
        self.image = None # reset the image

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
            (old_connection.key[0], new_node.id), 1.0)
        assert new_cx_1.key not in self.connection_genome.keys()
        self.connection_genome[new_cx_1.key] = new_cx_1

        new_cx_2 = Connection((new_node.id, old_connection.key[1]),
            old_connection.weight)
        assert new_cx_2.key not in self.connection_genome.keys()
        self.connection_genome[new_cx_2.key] = new_cx_2

        self.update_node_layers() # update the layers of the nodes
        self.image = None # reset the image
        
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
        self.image = None # reset the image

    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections())
        if len(eligible_cxs) < 1:
            return
        cx = random_choice(eligible_cxs, 1, False)[0]
        cx.enabled = False
        self.image = None # reset the image

    def update_node_layers(self) -> int:
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
        if self.config.use_radial_distance:
            # d = sqrt(x^2 + y^2)
            inputs.append(math.sqrt(inputs[0]**2 + inputs[1]**2))
        if self.config.use_input_bias:
            inputs.append(1.0)  # bias = 1.0

        for inp, node in zip(inputs, self.input_nodes().values()):
            # inputs are first N nodes
            node.sum_input = inp
            node.output = node.activation(inp)

    def get_layer(self, layer_index) -> list:
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
        for _, cx in self.connection_genome.items():
            if cx.weight < self.config.weight_threshold and cx.weight >\
                 -self.config.weight_threshold:
                cx.weight = torch.tensor(0.0)
            if not isinstance(cx.weight, torch.Tensor):
                cx.weight = torch.tensor(cx.weight)
            cx.weight = torch.clamp(cx.weight, -self.config.max_weight, self.config.max_weight)
            

    def reset_activations(self):
        """Resets all node activations to zero."""
        for _, node in self.node_genome.items():
            node.sum_inputs = torch.ones((self.config.res_h, self.config.res_w), device=self.device)/2.0
            node.outputs = torch.ones((self.config.res_h, self.config.res_w), device=self.device)/2.0

    def eval(self, inputs):
        """Evaluates the CPPN."""
        self.set_inputs(inputs)
        return self.feed_forward()

    def feed_forward(self):
        """Feeds forward the network."""
        if self.config.allow_recurrent:
            for _, input_node in self.input_nodes().items():
                # input nodes (handle recurrent)
                input_node.activate(get_incoming_connections(self, input_node), self.node_genome)

        # always an output node
        output_layer = self.get_output_layer_index()

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node.sum_input = 0
                node.outputs = 0
                node.activate(get_incoming_connections(self, node), self.node_genome)
                # node.outputs = np.clip(node.outputs, -1, 1) # clip output

        return [node.outputs for node in self.output_nodes()]

    def get_image(self, force_recalculate=False, override_h=None, override_w=None):
        """Returns an image of the network."""
        # apply size override
        if override_h is not None:
            self.config.res_h = override_h
        if override_w is not None:
            self.config.res_w = override_w

        # decide if we need to recalculate the image
        recalculate = False
        recalculate = recalculate or force_recalculate
        if isinstance(self.image, torch.Tensor):
            recalculate = recalculate or self.config.res_h == self.image.shape[0]
            recalculate = recalculate or self.config.res_w == self.image.shape[1]
        else:
            # no cached image
            recalculate = True

        if not recalculate:
            # return the cached image
            return self.image

        if self.config.allow_recurrent:
            # pixel by pixel (good for debugging/recurrent)
            self.image = self.get_image_data_serial()
        else:
            # whole image at once (100x faster)
            self.image = self.get_image_data_parallel()
        return self.image

    def get_image_data_serial(self):
        """Evaluate the network to get image data by processing each pixel
        serially. Much slower than the parallel method, but required if the
        network has recurrent connections."""
        res_h, res_w = self.config.res_h, self.config.res_w
        pixels = []
        for x in torch.linspace(-.5, .5, res_w,device=self.device):
            for y in torch.linspace(-.5, .5, res_h,device=self.device):
                outputs = self.eval([x, y])
                pixels.extend(outputs)
        if len(self.config.color_mode)>2:
            pixels = torch.reshape(pixels, (res_w, res_h, self.n_outputs))
        else:
            pixels = torch.reshape(pixels, (res_w, res_h))

        self.image = pixels
        return pixels

    def get_image_data_parallel(self):
        """Evaluate the network to get image data in parallel"""
        res_h, res_w = self.config.res_h, self.config.res_w
        if CPPN.pixel_inputs is None or CPPN.pixel_inputs.shape != (res_h,res_w):
            # initialize inputs if the resolution changed
            CPPN.initialize_inputs(res_h, res_w,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.n_inputs,
                self.device)

        # reset the activations to 0 before evaluating
        self.reset_activations()

        layers = feed_forward_layers(self) # get layers
        layers.insert(0, self.input_nodes().keys()) # add input nodes as first layer
        for layer in layers:
            # iterate over layers
            for node_index, node_id in enumerate(layer):
                # iterate over nodes in layer
                node = find_node_with_id(self.node_genome, node_id) # the current node

                # find incoming connections
                node_inputs = get_incoming_connections(self, node)

                # initialize the node's sum_inputs
                if node.type == NodeType.INPUT:
                    starting_input = CPPN.pixel_inputs[:,:,node_index]
                else:
                    starting_input = torch.zeros((res_h, res_w),device=self.device)

                node.initialize_sum(starting_input)
                # initialize the sum_inputs for this node
                node.activate(node_inputs, self.node_genome)

        # collect outputs from the last layer
        outputs = torch.stack([node.outputs for node in self.output_nodes().values()])

        # reshape the outputs to image shape
        if len(self.config.color_mode)>2:
            outputs =  outputs.permute(1, 2, 0) # move color axis to end
        else:
            outputs = torch.reshape(outputs, (res_h, res_w))

        self.image = outputs

        self.normalize_image()

        return self.image

    def normalize_image(self):
        """Normalize from -1 through 1 to 0 through 255 and convert to ints"""
        self.image = 1.0 - torch.abs(self.image)
        max_value = torch.max(self.image)
        min_value = torch.min(self.image)
        image_range = max_value - min_value
        self.image -= min_value
        if self.config.color_mode == 'HSL':
            self.image = hsv2rgb(self.image) # convert to RGB
        self.image *= 255
        if image_range != 0: # prevent divide by 0
            self.image /= image_range
        
        self.image = self.image.to(dtype=torch.uint8)

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
        difference_of_matching_weights = np.mean(
            difference_of_matching_weights)

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

    def update_with_fitness(self, fit, num_in_species):
        self.fitness = fit
        if(num_in_species > 0):
            self.adjusted_fitness = self.fitness / num_in_species  # local competition
            try:
                assert not torch.isnan(self.adjusted_fitness).any(), f"adjusted fitness was nan: fit: {self.fitness} n_in_species: {num_in_species}"
            except AssertionError as e:
                print(e)
        else:
            self.adjusted_fitness = self.fitness
            print("ERROR: num_in_species was 0")


    def crossover(self, other):
        """ Configure a new genome by crossover from two parent genomes. """
        child = CPPN(self.config, {}, {}) # create an empty child genome
        
        # determine which parent is more fit
        if self.fitness and other.fitness:        
            if self.fitness > other.fitness:
                parent1, parent2 = self, other
            else:
                parent1, parent2 = other, self
        else:
            # fitness undefined, choose randomly
            if torch.rand() > 0.5:
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

        child.update_node_layers()
        return child

    def crossover_(self, other_parent):
        """Crossover with another CPPN using the method in Stanley and Miikkulainen (2007)."""
        child = CPPN(self.config) # create child

        # disjoint/excess genes are inherited from first parent
        child.node_genome = copy.deepcopy(self.node_genome)
        child.connection_genome = copy.deepcopy(self.connection_genome)

        # line up by innovation number and find matches
        child.connection_genome.sort(key=lambda x: x.innovation)
        matching1, matching2 = get_matching_connections(
            self.connection_genome, other_parent.connection_genome)

        for match_1, match_2 in zip(matching1, matching2):
            child_cx = child.connection_genome[[x.innovation\
                for x in child.connection_genome].index(
                match_1.innovation)]

            # Matching genes are inherited randomly
            inherit_from_parent_1 = random_uniform() < .5
            if inherit_from_parent_1:
                child_cx.weight = match_1.weight
                new_from = copy.deepcopy(match_1.from_node)
                new_to = copy.deepcopy(match_1.to_node)
            else:
                child_cx.weight = match_2.weight
                new_from = copy.deepcopy(match_2.from_node)
                new_to = copy.deepcopy(match_2.to_node)

            # assign new nodes and connections
            child_cx.from_node = new_from
            child_cx.to_node = new_to
            try:
                existing = find_node_with_id(child.node_genome, new_from.id)
                index_existing = child.node_genome.index(existing)
                child.node_genome[index_existing] = new_from
            except ValueError:
                # this node does not exist in the child genome, don't add connection
                child.connection_genome.remove(child_cx)
                raise ValueError(f"Could not find node with id {new_from.id}")
            try:
                existing = find_node_with_id(child.node_genome, new_to.id)
                index_existing = child.node_genome.index(existing)
                child.node_genome[index_existing] = new_to
            except ValueError:
                # this node does not exist in the child genome, don't add connection
                child.connection_genome.remove(child_cx)
                raise ValueError(f"Could not find node with id {new_to.id}")

            if(not match_1.enabled or not match_2.enabled):
                if random_uniform() < 0.75:  # 0.75 from Stanley/Miikulainen 2007
                    child_cx.enabled = False

        # child.update_node_layers()

        return child
