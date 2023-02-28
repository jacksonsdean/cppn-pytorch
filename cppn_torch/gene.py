from enum import IntEnum
import json
from typing import Callable
import torch
from cppn_torch.activation_functions import identity

from cppn_torch.graph_util import name_to_fn

class NodeType(IntEnum):
    """Enum for the type of node."""
    INPUT  = 0
    OUTPUT = 1
    HIDDEN = 2

class Gene(object):
    """Represents either a node or connection in the CPPN"""
    def __init__(self, key=None) -> None:
        pass
    def copy(self):
        new_gene = self.__class__(key=self.key)
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
        
        assert isinstance(self.key, tuple), "Cannot crossover genes with non-tuple keys"
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
        i = Node.empty()
        i = i.from_json(json_dict)
        return i

    @staticmethod
    def empty():
        """Returns an empty node. Default activation function is identity."""
        return Node(0, identity, NodeType.HIDDEN, 0)

    def __init__(self, key, activation=None, _type=2, _layer=999, node_agg="sum") -> None:
        self.activation = activation
        self.id = key
        self.type = _type
        self.layer = _layer
        self.sum_inputs = None
        self.outputs = None
        self.agg = node_agg
        super().__init__()
    
    @property
    def key(self):
        return self.id
    @property
    def _gene_attributes(self):
        return [('activation', self.activation), ('type', self.type)]
    
    def to_cpu(self):
        if self.sum_inputs is not None:
            self.sum_inputs = self.sum_inputs.cpu()
        if self.outputs is not None:
            self.outputs = self.outputs.cpu()

    def activate(self, X, W):
        """Activates the node given a list of connections that end here."""
        assert isinstance(self.activation, Callable), "activation function is not a function"
        if X is None:
            return
        if W is None:
            self.outputs = self.activation(X)
            return

        if self.agg == 'sum':
            # self.sum_inputs = torch.matmul(W, X) # slower for small matrices (?)
            for x, w in zip(X, W):
                self.sum_inputs += x*w
            
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
        if isinstance(self.activation, Callable):
            self.activation = self.activation.__name__
    
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
        assert isinstance(self.weight, torch.Tensor), "weight is not a tensor"
        self.weight = float(self.weight)
    def deserialize(self):
        pass
    
    def to_cpu(self):
        assert isinstance(self.weight, torch.Tensor), "weight is not a tensor"
        self.weight = self.weight.cpu()

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
        i = Connection((-1,-1), 0)
        i.from_json(json_dict)
        i.weight = torch.tensor(i.weight)
        return i

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"([{self.key[0]}->{self.key[1]}] "+\
            f"W:{self.weight:3f} E:{int(self.enabled)} R:{int(self.is_recurrent)})"

