from enum import IntEnum
import json
from typing import Callable
import torch
from torch import nn
from copy import deepcopy
from cppn_torch.activation_functions import identity

from cppn_torch.graph_util import name_to_fn

class NodeType(IntEnum):
    """Enum for the type of node."""
    INPUT  = 0
    OUTPUT = 1
    HIDDEN = 2

class Gene(nn.Module):
    """Represents either a node or connection in the CPPN"""
    def __init__(self, key=None) -> None:
        super().__init__()
        self.key_ = key
        assert isinstance(key, str), f"key must be a string, not {type(key)}"
        
    def copy(self,deep=False):
        new_gene = self.__class__(key=self.key)
        for name, value in self._gene_attributes:
            if deep:
                setattr(new_gene, name, deepcopy(value))
            else:
                setattr(new_gene, name, value)

        return new_gene
    
    def mutate(self):
        raise NotImplementedError

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
        
        # assert isinstance(self.key, tuple), f"Cannot crossover genes with non-tuple keys, has type {type(self.key)} key: {self.key}"
        new_gene = self.__class__(self.key)

        for name, value in self._gene_attributes:
            if torch.rand(1)[0] < 0.5:
                setattr(new_gene, name, value)
            else:
                setattr(new_gene, name, getattr(other, name))

        return new_gene
    
class Node(Gene):
    """Represents a node in the CPPN."""
    # TODO: aggregation function, response(?)
    
    @staticmethod
    def create_from_json(json_dict):
        """Constructs a node from a json dict or string."""
        i = Node.empty()
        i = i.from_json(json_dict)
        return i

    @staticmethod
    def empty():
        """Returns an empty node. Default activation function is identity."""
        return Node("0", identity, NodeType.HIDDEN, 0)

    def __init__(self, key, activation=None, _type=2, _layer=999, node_agg="sum", device="cpu", grad=True) -> None:
        super().__init__(key)
        self.device = device
        self.activation = activation
        self.set_activation(activation)
        self.id = key
        self.type = _type
        self.layer = _layer
        self.sum_inputs = None
        self.outputs = None
        self.agg = node_agg
        self.bias = nn.Parameter(torch.zeros(1, device=device, requires_grad=grad))
        # self.bias = torch.zeros(1, device=device, requires_grad=grad)
        self.activation_params = []
    
    @property
    def key(self):
        return self.id
    @property
    def _gene_attributes(self):
        return [('activation', self.activation), ('type', self.type), ('bias', self.bias)]
    
    def set_activation(self, activation):
        self.activation = activation()
        self.activation_params = []

   
    def params(self):
        return [self.bias] + self.activation_params

    def activate(self, X, W):
        """Activates the node given a list of connections that end here."""
        assert isinstance(self.activation, Callable), f"activation function <{self.activation}> is not a function"

        if X is None:
            return
        if W is None:
            self.outputs = self.activation(X) + self.bias
            return
        
        # X_shape = (h,w,c)
        # W_shape = (c)
        
        if self.agg == 'sum':
            self.sum_inputs =torch.matmul(X, W) # (h,w,c) * (c) = (h,w)
            self.sum_inputs += self.bias
            
        else:
            X_shape = X.shape[1:]
            X = X.reshape(X.shape[0], -1)
            W = W.unsqueeze(1) # reshape for broadcasting
            weighted_x = torch.mul(W, X)
            
            if self.agg == 'mean':
                self.sum_inputs = weighted_x.mean(dim=0)
            elif self.agg == 'max':
                self.sum_inputs = weighted_x.max(dim=0)[0]
            elif self.agg == 'min':
                self.sum_inputs = weighted_x.min(dim=0)[0] 
            else:
                raise ValueError(f"Unknown aggregation function {self.agg}")
            
            self.sum_inputs = self.sum_inputs.reshape(X_shape) # reshape back to original shape
        
        self.sum_inputs += self.bias
        if not isinstance(self.activation, torch.nn.Conv2d):
            self.outputs = self.activation(self.sum_inputs)  # apply activation
        else:
            self.outputs = self.activation(self.sum_inputs.unsqueeze(0)).squeeze(0)

    def initialize_sum(self, initial_sum):
        """Activates the node."""
        self.sum_inputs = initial_sum

    def serialize(self):
        """Makes the node serializable."""
        return 
        self.type = self.type.value if isinstance(self.type, NodeType) else self.type
        self.id = int(self.id)
        self.layer = int(self.layer)
        self.sum_inputs = None
        self.outputs = None
        # self.bias = self.bias.item() if isinstance(self.bias, torch.Tensor) else self.bias
        self.device = str(self.device)
        if not hasattr(self.activation, '__name__'):
            self.activation = self.activation.__class__.__name__
        else:
            self.activation = self.activation.__name__
    
    def deserialize(self):
        """Makes the node functional"""
        self.sum_inputs = None
        self.sum_inputs = None
        self.activation = name_to_fn(self.activation) if isinstance(self.activation, str) else self.activation
        self.type = NodeType(self.type)
        self.device = torch.device(self.device)
        for name, value in self._gene_attributes:
            if isinstance(value,float):
                # convert to tensor
                setattr(self, name, torch.tensor(value))

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
    
    def to(self, device):
        self.bias = self.bias.to(device)
        self.device = device
        if self.sum_inputs is not None:
            self.sum_inputs = self.sum_inputs.to(device)
        if self.outputs is not None:
            self.outputs = self.outputs.to(device)
        if isinstance(self.activation, torch.nn.Conv2d):
            self.activation = self.activation.to(device)
        return self

class Connection(Gene):
    """
    Represents a connection between two nodes.

    where innovation number is the same for all of same connection
    i.e. 2->5 and 2->5 have same innovation number, regardless of individual
    """
    def __init__(self, key, weight = None, enabled = True) -> None:
        super().__init__(key)
        
        # Initialize
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        self.weight = nn.Parameter(weight)
        # self.innovation = Connection.get_innovation(key)
        self.enabled = enabled
        # self.is_recurrent = to_node.layer < from_node.layer
        self.is_recurrent = False # TODO
        
    @property
    def key(self):
        return self.key_
    @property
    def key_tuple(self):
        return self.key_.split(',')
    @property
    def from_node(self):
        return self.key_[0]
    @property
    def to_node(self):
        return self.key_[1]
    @property
    def _gene_attributes(self):
        return [('weight', self.weight), ('enabled', self.enabled)]
    
    def serialize(self):
        assert isinstance(self.weight, torch.Tensor), "weight is not a tensor"
        self.weight = float(self.weight)
    def deserialize(self):
        pass
    
    def to_json(self):
        """Converts the connection to a json string."""
        return json.dumps(self.__dict__)

    def from_json(self, json_dict):
        """Constructs a connection from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        self.__dict__ = json_dict
        return self

    @staticmethod
    def create_from_json(json_dict):
        """Constructs a connection from a json dict or string."""
        i = Connection(str((-1,-1)), 0)
        i.from_json(json_dict)
        i.weight = torch.tensor(i.weight)
        return i

    # def __str__(self):
        # return self.__repr__()
    # def __repr__(self):
        # return f"([{self.key.split(',')[0]}->{self.key.split(',')[1]}] "+\
            # f"W:{self.weight.item():3f} E:{int(self.enabled)} R:{int(self.is_recurrent)})"
    
    # def to(self, device):
        # self.weight = self.weight.to(device)
        # return self
