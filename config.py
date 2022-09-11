"""Stores configuration parameters for the CPPN."""
import inspect
import json
import random
import sys
from typing import Callable

import torch

from cppn_neat.activation_functions import identity

try:
    from activation_functions import get_all
    from graph_util import name_to_fn
except ModuleNotFoundError:
    from cppn_neat.activation_functions import get_all
    from cppn_neat.graph_util import name_to_fn

class   Config:
    """Stores configuration parameters for the CPPN."""
    # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        # Initialize to default values
        # These are only used if the frontend client doesn't override them
        self.target = None
        self.population_size = 10
        self.num_generations = 1000
        self.species_target = 3
        self.population_elitism = 1
        self.within_species_elitism = 1 #TODO NOT SURE IF WORKS can converge too quickly on small populations
        self.res_w = 28
        self.res_h = 28
        self.save_w = 512
        self.save_h = 512
        self.color_mode = "RGB"
        self.do_crossover = True
        self.crossover_ratio = .75 # from original NEAT
        self.use_dynamic_mutation_rates = True
        self.dynamic_mutation_rate_end_modifier = 0.1
        self.allow_recurrent = False
        self.init_connection_probability = 0.85
        self.activations = get_all()
        self.seed = random.randint(0, 100000)
        self.device = "cpu"
        self.normalize_outputs = True
        self.genome_type = None
        
        self.novelty_selection_ratio_within_species = 0
        self.novelty_adjusted_fitness_proportion = 0
        
        # self.fitness_function = 'test' # should get all white pixels
        # self.fitness_function = 'xor' # for debugging
        # self.fitness_function = 'mse' # default -mse
        self.fitness_function = 'average' # all fitness functions are averaged
        # self.fitness_function = 'haarpsi' # perceptual similarity
        self.fitness_schedule_type = "alternating"
        self.fitness_schedule_period = 10
        # self.fitness_schedule = ["mse", "haarpsi", "ssim", "psnr", "fsim"]
        # self.fitness_schedule = ["mse", "psnr"]
        self.fitness_schedule = None
        self.min_fitness = None
        self.max_fitness = None
        
        # NEAT specific parameters
        self.use_speciation = True
        self.init_species_threshold = 3
        self.species_threshold_delta = .1
        self.species_stagnation_threshold = 100
        self.species_selection_ratio = .8 # truncation selection within species
        self.crossover_between_species_probability = 0.001 # .001 in the original NEAT

        """DGNA: the probability of adding a node is 0.5 and the
        probability of adding a connection is 0.4.
        SGNA: probability of adding a node is 0.05 and the
         probability of adding a connection is 0.04.
        NEAT: probability of adding a node is 0.03 and the
          probability of adding a connection is 0.05."""
        self.prob_mutate_activation = .5
        self.prob_mutate_weight = .80 # .80 in the original NEAT
        self.prob_add_connection = .25 # 0.05 in the original NEAT
        self.prob_add_node = .65 # 0.03 in original NEAT
        self.prob_remove_node = 0.15
        self.prob_disable_connection = .015

        self.max_weight = 3.0
        self.weight_threshold = 0
        self.prob_random_restart =.001
        self.prob_weight_reinit = 0.1 * .80 # .1 in the original NEAT (.1 of .8)
        self.prob_reenable_connection = 0.1
        self.weight_mutation_max = 2
        
        self.output_activation = None
        
        self.output_dir = None

        # DGNA/SGMA uses 1 or 2 so that patterns in the initial
        # generation would be nontrivial (Stanley, 2007).
        # Original NEAT paper uses 0
        self.hidden_nodes_at_start = 0

        self.allow_input_activation_mutation = True

        self.animate = False

        # https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf page 148
        self.use_input_bias = True # SNGA,
        # self.use_input_bias = False # SNGA,
        # self.use_radial_distance = True # bias towards radial symmetry
        self.use_radial_distance = False # bias towards radial symmetry
        
        self.num_inputs = 2
        self.num_outputs = len(self.color_mode)
        if self.use_input_bias:
            self.num_inputs += 1
        if self.use_radial_distance:
            self.num_inputs += 1
            
        # MAP-Elites
        self.map_elites_resolution = [6,6]
        self.map_elites_max_values = [15000,30]
        self.map_elites_min_values = [100,9]
            

        self.novelty_archive_len = 20
        self.novelty_k = 5
        self.autoencoder_frequency = 10
            

        
    def apply_condition(self, key, value):
        """Applies an experimental condition to the configuration."""
        setattr(self, key, value)
        
    #################    
    # Serialization #
    #################
    
    def serialize(self):
        self.fns_to_strings()
        
    def deserialize(self):
        self.strings_to_fns()

    def fns_to_strings(self):
        """Converts the activation functions to strings."""
        if self.genome_type:
            self.genome_type = self.genome_type.__name__
        
        self.activations= [fn.__name__ if not isinstance(fn, str) else fn for fn in self.activations]
        if isinstance(self.fitness_function, Callable):
            self.fitness_function = self.fitness_function.__name__
        if self.fitness_schedule is not None:
            for i, fn in enumerate(self.fitness_schedule):
                if isinstance(fn, Callable):
                    self.fitness_schedule[i] = fn.__name__
        
        if self.output_activation is None:
            self.output_activation = ""
        else:
            self.output_activation = self.output_activation.__name__ if\
                not isinstance(self.output_activation, str) else self.output_activation
    
    def strings_to_fns(self):
        """Converts the activation functions to functions."""
        if self.genome_type:
            found = False
            modules = sys.modules
            for m in modules:
                try:
                    for c in inspect.getmembers(m, inspect.isclass):
                        if c[0] == self.genome_type:
                            self.genome_type = c[1]
                            found = True
                            break
                    if found:
                        break
                except:
                    continue
                
            
        self.activations = [name_to_fn(name) if isinstance(name, str) else name for name in self.activations ]
        
        try:
            self.fitness_function = name_to_fn(self.fitness_function)
        except ValueError:
            self.fitness_function = None
        self.output_activation = name_to_fn(self.output_activation) if isinstance(self.output_activation, str) else self.output_activation
        
        if self.fitness_schedule is not None:
            for i, fn in enumerate(self.fitness_schedule):
                if isinstance(fn, str):
                    self.fitness_schedule[i] = name_to_fn(fn)

    def to_json(self):
        """Converts the configuration to a json string."""
        self.fns_to_strings()
        json_string = json.dumps(self.__dict__, sort_keys=True, indent=4)
        self.strings_to_fns()
        return json_string


    def from_json(self, json_dict):
        """Converts the configuration from a json string."""
        if isinstance(json_dict, dict):
            json_dict = json.loads(json_dict)
        self.fns_to_strings()
        for key, value in json_dict.items():
            setattr(self, key, value)
        self.strings_to_fns()
        
    def save(self, filename):
        """Saves the configuration to a file."""
        with open(filename, "w") as f:
            f.write(self.to_json())
            f.close()

    @staticmethod
    def create_from_json(json_str):
        """Creates a configuration from a json string."""
        if isinstance(json_str, str):
            json_str = json.loads(json_str)
        config = Config()
        for key, value in json_str.items():
            setattr(config, key, value)
        config.strings_to_fns()
        return config
