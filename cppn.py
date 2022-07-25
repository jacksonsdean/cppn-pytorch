"""Contains the CPPN, Node, and Connection classes."""
import copy
from enum import IntEnum
import math
import json
from typing import Callable
# import numpy as np
import torch
try:
    from activation_functions import identity
    from graph_util import name_to_fn, choose_random_function, is_valid_connection
    from graph_util import get_matching_connections, find_node_with_id
    from graph_util import get_incoming_connections, feed_forward_layers
    from graph_util import hsv2rgb
except ModuleNotFoundError:
    from cppn_neat.activation_functions import identity
    from cppn_neat.graph_util import get_matching_connections, find_node_with_id
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

class Node:
    """Represents a node in the CPPN."""

    @staticmethod
    def create_from_json(json_dict):
        """Constructs a node from a json dict or string."""
        i = Node(None, None, None, None)
        i = i.from_json(json_dict)
        return i

    @staticmethod
    def empty():
        """Returns an empty node."""
        return Node(identity, NodeType.HIDDEN, 0, 0)

    def __init__(self, activation, _type, _id, _layer=999) -> None:
        self.activation = activation
        self.id = _id
        self.type = _type
        self.layer = _layer
        self.sum_inputs = None
        self.outputs = None

    def activate(self, incoming_connections):
        """Activates the node given a list of connections that end here."""
        for cx in incoming_connections:
            if cx.from_node.outputs is not None:
                inputs = cx.from_node.outputs * cx.weight
                self.sum_inputs = self.sum_inputs + inputs
        
        self.outputs = self.activation(self.sum_inputs)  # apply activation

    def initialize_sum(self, initial_sum):
        """Activates the node."""
        self.sum_inputs = initial_sum

    def serialize(self):
        """Makes the node serializable."""
        self.type = self.type.value if isinstance(self.type, NodeType) else self.type
        self.id = int(self.id)
        self.layer = int(self.id)
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

class Connection:
    """
    Represents a connection between two nodes.

    where innovation number is the same for all of same connection
    i.e. 2->5 and 2->5 have same innovation number, regardless of individual
    """
    innovations = []

    @staticmethod
    def get_innovation(from_node, to_node):
        """Returns the innovation number for the connection."""
        connection_from_to = (from_node.id, to_node.id) # based on id

        if connection_from_to in Connection.innovations:
            return Connection.innovations.index(connection_from_to)

        Connection.innovations.append(connection_from_to)
        return len(Connection.innovations) - 1

    def __init__(self, from_node, to_node, weight, enabled = True) -> None:
        # Initialize
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.innovation = Connection.get_innovation(from_node, to_node)
        self.enabled = enabled
        self.is_recurrent = to_node.layer < from_node.layer

    def serialize(self):
        self.weight = float(self.weight)
    def deserialize(self):
        pass
    
    def to_json(self):
        """Converts the connection to a json string."""
        self.innovation = int(self.innovation)
        if isinstance(self.from_node, Node):
            self.from_node = self.from_node.to_json()
        if isinstance(self.to_node, Node):
            self.to_node = self.to_node.to_json()
        return json.dumps(self.__dict__)

    def from_json(self, json_dict):
        """Constructs a connection from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        self.__dict__ = json_dict
        self.from_node = Node.create_from_json(self.from_node)
        self.to_node = Node.create_from_json(self.to_node)
        return self

    @staticmethod
    def create_from_json(json_dict):
        """Constructs a connection from a json dict or string."""
        f_node = Node.empty()
        t_node = Node.empty()
        i = Connection(f_node, t_node, 0)
        i.from_json(json_dict)
        return i

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"([{self.from_node.id}->{self.to_node.id}]"+\
            f"I:{self.innovation} W:{self.weight:3f} E:{self.enabled} R:{self.is_recurrent})"


class CPPN():
    """A CPPN Object with Nodes and Connections."""

    pixel_inputs = torch.zeros((0, 0, 0)) # (res_h, res_w, n_inputs)
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

    def __init__(self, config, nodes = None, connections = None, device=None) -> None:
        self.device = device
        if self.device is None:
            # no device specified, try to use GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.image = None
        self.node_genome = []  # inputs first, then outputs, then hidden
        self.connection_genome = []
        self.selected = False
        
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

    def initialize_connection_genome(self):
        """Initializes the connection genome."""
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(0, output_layer):
            layer_from = self.get_layer(layer_index)
            for _, from_node in enumerate(layer_from):
                layer_to = self.get_layer(layer_index+1)
                for _, to_node in enumerate(layer_to):
                    new_cx = Connection(
                        from_node, to_node, self.random_weight())
                    self.connection_genome.append(new_cx)
                    if torch.rand(1)[0] > self.config.init_connection_probability:
                        new_cx.enabled = False

    def initialize_node_genome(self):
        """Initializes the node genome."""
        total_node_count = self.n_inputs + \
            self.n_outputs + self.config.hidden_nodes_at_start
        for _ in range(self.n_inputs):
            self.node_genome.append(
                Node(identity, NodeType.INPUT, self.get_new_node_id(), 0))
        for _ in range(self.n_inputs, self.n_inputs + self.n_outputs):
            if self.config.output_activation is None:
                output_fn = choose_random_function(self.config)
            else:
                output_fn = self.config.output_activation
            self.node_genome.append(
                Node(output_fn, NodeType.OUTPUT, self.get_new_node_id(), 2))
        for _ in range(self.n_inputs + self.n_outputs, total_node_count):
            self.node_genome.append(Node(choose_random_function(self.config), NodeType.HIDDEN,
                self.get_new_node_id(), 1))
    
    def serialize(self):
        del CPPN.pixel_inputs
        CPPN.pixel_inputs = None
        if self.image is not None:
            self.image = self.image.cpu().numpy().tolist() if\
                isinstance(self.image, torch.Tensor) else self.image
        for node in self.node_genome:
            node.serialize()
        for connection in self.connection_genome:
            connection.serialize()
        self.config.serialize()

    def deserialize(self):
        for node in self.node_genome:
            node.deserialize()
        for connection in self.connection_genome:
            connection.deserialize()
        self.config.deserialize()
        
    def to_json(self):
        """Converts the CPPN to a json dict."""
        self.serialize()
        img = json.dumps(self.image) if self.image is not None else None
        # make copies to keep the CPPN intact
        copy_of_nodes = copy.deepcopy(self.node_genome)
        copy_of_connections = copy.deepcopy(self.connection_genome)
        return {"node_genome": [n.to_json() for n in copy_of_nodes], "connection_genome":\
            [c.to_json() for c in copy_of_connections], "image": img, "selected": self.selected}

    def from_json(self, json_dict):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        for key, value in json_dict.items():
            setattr(self, key, value)
       
       # connections
        for i, cx in enumerate(self.connection_genome):
            self.connection_genome[i] = Connection.create_from_json(cx) if\
                isinstance(cx, (dict,str)) else cx
            assert isinstance(self.connection_genome[i], Connection),\
                f"Connection is a {type(self.connection_genome[i])}: {self.connection_genome[i]}"
       
       # nodes
        for i, n in enumerate(self.node_genome):
            self.node_genome[i] = Node.create_from_json(n) if\
                isinstance(n, (dict,str)) else n

            assert isinstance(self.node_genome[i], Node),\
                f"Node is a {type(self.node_genome[i])}: {self.node_genome[i]}"
        
        # make sure references are correct
        for cx in self.connection_genome:
            cx.from_node = find_node_with_id(self.node_genome, cx.from_node.id)
            cx.to_node = find_node_with_id(self.node_genome, cx.to_node.id)

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

    def get_new_node_id(self):
        """Returns a new node id."""
        new_id = 0
        while len(self.node_genome) > 0 and new_id in [node.id for node in self.node_genome]:
            new_id += 1
        return new_id

    def enabled_connections(self):
        """Returns a yield of enabled connections."""
        for connection in self.connection_genome:
            if connection.enabled:
                yield connection

    def mutate_activations(self):
        """Mutates the activation functions of the nodes."""
        eligible_nodes = list(self.hidden_nodes())
        if self.config.output_activation is None:
            eligible_nodes.extend(self.output_nodes())
        if self.config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes())
        for node in eligible_nodes:
            if random_uniform(0,1) < self.config.prob_mutate_activation:
                node.activation = choose_random_function(self.config)

    def mutate_weights(self):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """

        for connection in self.connection_genome:
            if random_uniform(0, 1) < self.config.prob_mutate_weight:
                connection.weight += random_uniform(-self.config.weight_mutation_max,
                                               self.config.weight_mutation_max, device=self.device)
            elif random_uniform(0, 1) < self.config.prob_weight_reinit:
                connection.weight = self.random_weight()

        self.clamp_weights()

    def mutate(self):
        """Mutates the CPPN based on its config."""
        if random_uniform(0,1) < self.config.prob_add_node:
            self.add_node()
        if random_uniform(0,1) < self.config.prob_remove_node:
            self.remove_node()
        if random_uniform(0,1) < self.config.prob_add_connection:
            self.add_connection()
        if random_uniform(0,1) < self.config.prob_disable_connection:
            self.disable_connection()

        self.mutate_activations()
        self.mutate_weights()
        self.update_node_layers()
        self.disable_invalid_connections()

    def disable_invalid_connections(self):
        """Disables connections that are not compatible with the current configuration."""
        for connection in self.connection_genome:
            if connection.enabled:
                if not is_valid_connection(connection.from_node, connection.to_node, self.config):
                    connection.enabled = False


    def add_connection(self):
        """Adds a connection to the CPPN."""
        for _ in range(20):  # try 20 times max
            [from_node, to_node] = random_choice(
                self.node_genome, 2, replace=False)

            # look to see if this connection already exists
            existing_cx = None
            for cx in self.connection_genome:
                if cx.from_node == from_node and cx.to_node == to_node:
                    existing_cx = cx

            # if it does exist and it is disabled, there is a chance to reenable
            if existing_cx is not None:
                if not existing_cx.enabled:
                    if random_uniform() < self.config.prob_reenable_connection:
                        existing_cx.enabled = True # re-enable the connection
                break  # don't allow duplicates, don't enable more than one connection

            # else if it doesn't exist, check if it is valid
            if is_valid_connection(from_node, to_node, self.config):
                # valid connection, add
                new_cx = Connection(from_node, to_node, self.random_weight())
                self.connection_genome.append(new_cx)
                self.update_node_layers()
                break # found a valid connection, don't add more than one

            # else failed to find a valid connection, don't add and try again

    def add_node(self):
        """Adds a node to the CPPN.
            Looks for an eligible connection to split, add the node in the middle
            of the connection.
        """
        # only add nodes in the middle of non-recurrent connections
        eligible_cxs = [
            cx for cx in self.connection_genome if not cx.is_recurrent]

        if len(eligible_cxs) == 0:
            return # there are no eligible connections, don't add a node

        # choose a random eligible connection
        old_connection = random_choice(eligible_cxs,1,replace=False)[0]

        # create the new node
        new_node = Node(choose_random_function(self.config),
                        NodeType.HIDDEN, self.get_new_node_id(), 999)
        self.node_genome.append(new_node)  # add a new node between two nodes

        # disable old connection
        old_connection.enabled = False

        # The connection between the first node in the chain and the
        # new node is given a weight of one and the connection between
        # the new node and the last node in the chain
        # is given the same weight as the connection being split
        self.connection_genome.append(Connection(
            find_node_with_id(self.node_genome, old_connection.from_node.id),
            find_node_with_id(self.node_genome, new_node.id),
            1.0))

        self.connection_genome.append(Connection(
            find_node_with_id(self.node_genome, new_node.id),
            find_node_with_id(self.node_genome, old_connection.to_node.id),
            old_connection.weight))


        self.update_node_layers() # update the layers of the nodes

    def remove_node(self):
        """Removes a node from the CPPN.
            Only hidden nodes are eligible to be removed.
        """

        hidden = self.hidden_nodes()
        if len(hidden) == 0:
            return # no eligible nodes, don't remove a node

        # choose a random node
        node_id_to_remove = random_choice([n.id for n in hidden], 1, False)[0]

        for cx in self.connection_genome[::-1]:
            if node_id_to_remove in [cx.from_node.id, cx.to_node.id]:
                self.connection_genome.remove(cx)
        for node in self.node_genome[::-1]:
            if node.id == node_id_to_remove:
                self.node_genome.remove(node)
                break

        self.update_node_layers()
        self.disable_invalid_connections()

    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections())
        if len(eligible_cxs) < 1:
            return
        cx = random_choice(eligible_cxs, 1, False)[0]
        cx.enabled = False

    def update_node_layers(self) -> int:
        """Update the node layers."""
        layers = feed_forward_layers(self)

        for _, node in enumerate(self.input_nodes()):
            node.layer = 0
        for layer_index, layer in enumerate(layers):
            for _, node_id in enumerate(layer):
                node = find_node_with_id(self.node_genome, node_id)
                node.layer = layer_index + 1

    def input_nodes(self) -> list:
        """Returns a list of all input nodes."""
        return list(filter(lambda n: n.type == NodeType.INPUT, self.node_genome))

    def output_nodes(self) -> list:
        """Returns a list of all output nodes."""
        return list(filter(lambda n: n.type == NodeType.OUTPUT, self.node_genome))

    def hidden_nodes(self) -> list:
        """Returns a list of all hidden nodes."""
        return list(filter(lambda n: n.type == NodeType.HIDDEN, self.node_genome))

    def set_inputs(self, inputs):
        """Sets the input neurons outputs to the input values."""
        if self.config.use_radial_distance:
            # d = sqrt(x^2 + y^2)
            inputs.append(math.sqrt(inputs[0]**2 + inputs[1]**2))
        if self.config.use_input_bias:
            inputs.append(1.0)  # bias = 1.0

        for i, inp in enumerate(inputs):
            # inputs are first N nodes
            self.node_genome[i].sum_input = inp
            self.node_genome[i].output = self.node_genome[i].activation(inp)

    def get_layer(self, layer_index):
        """Returns a list of nodes in the given layer."""
        for node in self.node_genome:
            if node.layer == layer_index:
                yield node

    def get_layers(self):
        """Returns a dictionary of lists of nodes in each layer."""
        layers = {}
        for node in self.node_genome:
            if node.layer not in layers:
                layers[node.layer] = []
            layers[node.layer].append(node)
        return layers

    def clamp_weights(self):
        """Clamps all weights to the range [-max_weight, max_weight]."""
        for cx in self.connection_genome:
            if cx.weight < self.config.weight_threshold and cx.weight >\
                 -self.config.weight_threshold:
                cx.weight = 0
            if cx.weight > self.config.max_weight:
                cx.weight = self.config.max_weight
            if cx.weight < -self.config.max_weight:
                cx.weight = -self.config.max_weight

    def reset_activations(self):
        """Resets all node activations to zero."""
        for node in self.node_genome:
            node.sum_inputs = torch.ones((self.config.res_h, self.config.res_w), device=self.device)/2.0
            node.outputs = torch.ones((self.config.res_h, self.config.res_w), device=self.device)/2.0

    def eval(self, inputs):
        """Evaluates the CPPN."""
        self.set_inputs(inputs)
        return self.feed_forward()

    def feed_forward(self):
        """Feeds forward the network."""
        if self.config.allow_recurrent:
            for i in range(self.n_inputs):
                # input nodes (handle recurrent)
                for node_input in list(filter(lambda x,
                    index=i: x.to_node.id == self.node_genome[index].id,
                    self.enabled_connections())):
                    self.node_genome[i].sum_input +=\
                        node_input.from_node.outputs * node_input.weight

                self.node_genome[i].outputs =\
                    self.node_genome[i].activation(self.node_genome[i].sum_input)

        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node.sum_input = 0
                node.outputs = 0
                node_inputs = list(
                    filter(lambda x, n=node: x.to_node.id == n.id,
                        self.enabled_connections()))  # cxs that end here
                for cx in node_inputs:
                    node.sum_input += cx.from_node.outputs * cx.weight

                node.output = node.activation(node.sum_input)  # apply activation
                # node.output = np.clip(node.output, -1, 1) # clip output

        return [node.output for node in self.output_nodes()]

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
        layers.insert(0, [n.id for n in self.input_nodes()]) # add input nodes as first layer

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
                node.activate(node_inputs)

        # collect outputs from the last layer
        outputs = torch.stack([node.outputs for node in self.output_nodes()])

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

    def crossover(self, other_parent):
        """Crossover with another CPPN using the method in Stanley and Miikkulainen (2007)."""
        child = CPPN(self.config) # create child

        # disjoint/excess genes are inherited from first parent
        child.node_genome = copy.deepcopy(self.node_genome)
        child.connection_genome = copy.deepcopy(self.connection_genome)

        # line up by innovation number and find matches
        # child.connection_genome.sort(key=lambda x: x.innovation)
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
            try:
                existing = find_node_with_id(child.node_genome, new_to.id)
                index_existing = child.node_genome.index(existing)
                child.node_genome[index_existing] = new_to
            except ValueError:
                # this node does not exist in the child genome, don't add connection
                child.connection_genome.remove(child_cx)

            if(not match_1.enabled or not match_2.enabled):
                if random_uniform() < 0.75:  # 0.75 from Stanley/Miikulainen 2007
                    child_cx.enabled = False

        child.update_node_layers()

        return child
