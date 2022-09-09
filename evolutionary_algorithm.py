import copy
import math
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from cppn_neat.cppn import Node
import torch
from cppn_neat.graph_util import name_to_fn
from cppn_neat.util import get_avg_number_of_connections, get_avg_number_of_hidden_nodes, get_max_number_of_connections, visualize_network

class EvolutionaryAlgorithm(object):
    def __init__(self, target, config, debug_output=False) -> None:
        self.gen = 0
        self.next_available_id = 0
        self.debug_output = debug_output
        self.config = config
        Node.current_id =  self.config.num_inputs + self.config.num_outputs # reset node id counter
        self.show_output = True
        
        self.diversity_over_time = np.zeros(self.config.num_generations,dtype=float)
        self.population_over_time = np.zeros(self.config.num_generations,dtype=np.uint8)
        self.species_over_time = np.zeros(self.config.num_generations,dtype=np.float)
        self.species_threshold_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.nodes_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.connections_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.fitness_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.species_pops_over_time = []
        self.solutions_over_time = []
        self.species_champs_over_time = []
        self.time_elapsed = 0
        self.solution_generation = -1
        self.population = []
        self.solution = None
        self.this_gen_best = None
        
        self.run_number = 0
        
        self.solution_fitness = -math.inf
        self.best_genome = None
        self.genome_type = config.genome_type

        self.fitness_function = config.fitness_function
        
        if not isinstance(config.fitness_function, Callable):
            self.fitness_function = name_to_fn(config.fitness_function)
            self.fitness_function_normed = self.fitness_function
            
        self.target = target
    
    def get_mutation_rates(self):
        """Get the mutate rates for the current generation 
        if using a mutation rate schedule, else use config values

        Returns:
            float: prob_mutate_activation,
            float: prob_mutate_weight,
            float: prob_add_connection,
            float: prob_add_node,
            float: prob_remove_node,
            float: prob_disable_connection,
            float: weight_mutation_max, 
            float: prob_reenable_connection
        """
        if(self.config.use_dynamic_mutation_rates):
            run_progress = self.gen / self.config.num_generations
            end_mod = self.config.dynamic_mutation_rate_end_modifier
            prob_mutate_activation   = self.config.prob_mutate_activation   - (self.config.prob_mutate_activation    - end_mod * self.config.prob_mutate_activation)   * run_progress
            prob_mutate_weight       = self.config.prob_mutate_weight       - (self.config.prob_mutate_weight        - end_mod * self.config.prob_mutate_weight)       * run_progress
            prob_add_connection      = self.config.prob_add_connection      - (self.config.prob_add_connection       - end_mod * self.config.prob_add_connection)      * run_progress
            prob_add_node            = self.config.prob_add_node            - (self.config.prob_add_node             - end_mod * self.config.prob_add_node)            * run_progress
            prob_remove_node         = self.config.prob_remove_node         - (self.config.prob_remove_node          - end_mod * self.config.prob_remove_node)         * run_progress
            prob_disable_connection  = self.config.prob_disable_connection  - (self.config.prob_disable_connection   - end_mod * self.config.prob_disable_connection)  * run_progress
            weight_mutation_max      = self.config.weight_mutation_max      - (self.config.weight_mutation_max       - end_mod * self.config.weight_mutation_max)      * run_progress
            prob_reenable_connection = self.config.prob_reenable_connection - (self.config.prob_reenable_connection  - end_mod * self.config.prob_reenable_connection) * run_progress
            return  prob_mutate_activation, prob_mutate_weight, prob_add_connection, prob_add_node, prob_remove_node, prob_disable_connection, weight_mutation_max, prob_reenable_connection
        else:
            return  self.config.prob_mutate_activation, self.config.prob_mutate_weight, self.config.prob_add_connection, self.config.prob_add_node, self.config.prob_remove_node, self.config.prob_disable_connection, self.config.weight_mutation_max, self.config.prob_reenable_connection

    def evolve(self):
        raise NotImplementedError()
    def run_one_generation(self):
        raise NotImplementedError()
    
    def update_fitnesses_and_novelty(self):
        if self.show_output:
            pbar = trange(len(self.population))
        else:
            pbar = range(len(self.population))
        for i in pbar:
            if self.show_output:
                pbar.set_description_str("Evaluating gen " + str(self.gen) + ": ")
            
            if self.fitness_function.__name__ == "xor" or not self.fitness_function_normed:
                self.population[i].fitness = self.fitness_function(self.population[i])
            else:
                self.population[i].fitness = self.fitness_function_normed(self.population[i].get_image(), self.target)
            
            assert self.population[i].fitness >= 0, f"fitness must be non-negative for now, but got {self.population[i].fitness}"
        
        if self.show_output:
            pbar = trange(len(self.population))
        else:
            pbar = range(len(self.population))
            
        if(self.config.novelty_selection_ratio_within_species > 0 or self.config.novelty_adjusted_fitness_proportion > 0):
            # novelties = novelty_ae.get_ae_novelties(self.population)
            novelties = np.zeros(len(self.population))
            for i, n in enumerate(novelties):
                self.population[i].novelty = n
                self.novelty_archive = self.update_solution_archive(self.novelty_archive, self.population[i], self.config.novelty_archive_len, self.config.novelty_k)
    
    def update_solution_archive(self, solution_archive, genome, max_archive_length, novelty_k):
        # genome should already have novelty score
        # update existing novelty scores:
        # for i, archived_solution in enumerate(solution_archive):
        #     solution_archive[i].novelty = get_novelty(solution_archive, genome, novelty_k)
        solution_archive = sorted(solution_archive, reverse=True, key = lambda s: s.novelty)

        if(len(solution_archive) >= max_archive_length):
            if(genome.novelty > solution_archive[-1].novelty):
                # has higher novelty than at least one genome in archive
                solution_archive[-1] = genome # replace least novel genome in archive
        else:
            solution_archive.append(genome)
        return solution_archive
    
    def record_keeping(self):
        self.solution = self.population[0]
        self.this_gen_best = self.solution
        std_distance, avg_distance, max_diff = calculate_diversity_full(self.population)
        n_nodes = get_avg_number_of_hidden_nodes(self.population)
        n_connections = get_avg_number_of_connections(self.population)
        self.diversity_over_time[self.gen:] = avg_distance
        self.population_over_time[self.gen:] = float(len(self.population))
        self.nodes_over_time[self.gen:] = n_nodes
        self.connections_over_time[self.gen:] = n_connections

        # fitness
        if self.population[0].fitness > self.solution_fitness: # if the new parent is the best found so far
            self.solution = self.population[0]                 # update best solution records
            self.solution_fitness = self.solution.fitness
            self.solution_generation = self.gen
            self.best_genome = self.solution
            
                
        self.fitness_over_time[self.gen:] = self.solution_fitness # record the fitness of the current best over evolutionary time
        self.solutions_over_time.append(copy.deepcopy(self.solution))   
        
    def show_fitness_curve(self):
        # plt.close()
        plt.plot(self.fitness_over_time, label="Highest fitness")
        plt.title("Fitness over time")
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.legend()
        plt.show()
        
    def show_diversity_curve(self):
        # plt.close()
        plt.plot(self.diversity_over_time, label="Diversity")
        plt.title("Diversity over time")
        plt.ylabel("Diversity")
        plt.xlabel("Generation")
        plt.legend()
        plt.show()
       
    def mutate(self, child):
        rates = self.get_mutation_rates()
        child.fitness, child.adjusted_fitness = 0, 0 # new fitnesses after mutation
        child.mutate(rates)
    
    def get_best(self):
        max_fitness_individual = max(self.population, key=lambda x: x.fitness)
        return max_fitness_individual
    
    def print_best(self):
        best = self.get_best()
        print("Best:", best.id, best.fitness)
        
    def show_best(self):
        print()
        self.print_best()
        self.save_best_network_image()
        img = self.get_best().get_image().cpu().to(np.uint8).numpy()
        plt.imshow(img, cmap='gray')
        plt.show()
        
    def save_best_img(self, fname):
        img = self.get_best().get_image().cpu().numpy()
        plt.imsave(fname, img, cmap='gray')
        
        plt.close()
        img = self.this_gen_best.get_image().cpu().numpy()
        plt.imsave(fname.replace(".png","_final.png"), img, cmap='gray')

    def save_best_network_image(self):
        best = self.get_best()
        path = f"{self.config.output_dir}/genomes/best_{self.gen}.png"
        visualize_network(self.get_best(), sample=False, save_name=path, extra_text=f"Run {self.run_number} Generation: " + str(self.gen) + " fit: " + str(best.fitness) + " species: " + str(best.species_id))
    
     
    def print_fitnesses(self):
        div = calculate_diversity_full(self.population)
        print("Generation", self.gen, "="*100)
        print(f" |-Best: {self.get_best().id} ({self.get_best().fitness:.4f})")
        print(f" |  Average fitness: {torch.mean(torch.stack([i.fitness for i in self.population])):.7f} | adjusted: {torch.mean(torch.stack([i.adjusted_fitness for i in self.population])):.7f}")
        print(f" |  Diversity: std: {div[0]:.3f} | avg: {div[1]:.3f} | max: {div[2]:.3f}")
        print(f" |  Connections: avg. {get_avg_number_of_connections(self.population):.2f} max. {get_max_number_of_connections(self.population)}  | H. Nodes: avg. {get_avg_number_of_hidden_nodes(self.population):.2f} max: {get_max_number_of_hidden_nodes(self.population)}")
        for individual in self.population:
            print(f" |     Individual {individual.id} ({len(individual.hidden_nodes())}n, {len(list(individual.enabled_connections()))}c, s: {individual.species_id} fit: {individual.fitness:.4f}")
        
        print(f" Gen "+ str(self.gen), f"fitness: {self.get_best().fitness:.4f}")
        print()
        

def calculate_diversity_full(population):
    # very slow, compares every genome against every other
    diffs = []
    for i in population:
        for j in population:
            if i== j: continue
            diffs.append(i.genetic_difference(j))

    std_distance = np.std(diffs)
    avg_distance = np.mean(diffs)
    max_diff = np.max(diffs)if(len(diffs)>0) else 0
    return std_distance, avg_distance, max_diff


