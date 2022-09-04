import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from cppn_neat.cppn import *
from cppn_neat.util import *
from cppn_neat.species import *
import copy 

import random
import copy
import os

class NEAT():
    def __init__(self, target, config, debug_output=False, genome_type=CPPN) -> None:
        self.gen = 0
        self.next_available_id = 0
        self.debug_output = debug_output
        self.all_species = []
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
        self.species_threshold = self.config.init_species_threshold
        self.population = []
        self.solution = None
        
        self.solution_fitness = -math.inf
        self.best_genome = None

        self.genome_type = genome_type
        
        self.fitness_function = config.fitness_function
        
        if not isinstance(config.fitness_function, Callable):
            self.fitness_function = name_to_fn(config.fitness_function)
        
        # normalize fitness function
        self.update_fitness_function()
    
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

    def update_fitness_function(self):
        """Normalize fitness function if using a normalized fitness function"""
        if self.config.fitness_schedule is not None:
            if self.config.fitness_schedule_type == 'alternating':
                if self.gen==0:
                    self.fitness_function = self.config.fitness_schedule[0]
                elif self.gen % self.config.fitness_schedule_period == 0:
                        self.fitness_function = self.config.fitness_schedule[self.gen // self.config.fitness_schedule_period % len(self.config.fitness_schedule)]
                if self.debug_output:
                    print('Fitness function:', self.fitness_function.__name__)
            else:
                raise Exception("Unrecognized fitness schedule")
            
        if self.config.min_fitness is not None and self.config.max_fitness is not None:
            self.fitness_function_normed = lambda x,y: (self.config.fitness_function(x,y) - self.config.min_fitness) / (self.config.max_fitness - self.config.min_fitness)
        else:
            self.fitness_function_normed = self.fitness_function # no normalization

    def update_fitnesses_and_novelty(self):
        if self.show_output:
            pbar = trange(len(self.population))
        else:
            pbar = range(len(self.population))
        for i in pbar:
            if self.show_output:
                pbar.set_description_str("Evaluating gen " + str(self.gen) + ": ")
            
            if self.fitness_function.__name__ == "xor":
                self.population[i].fitness = self.fitness_function(self.population[i])
            else:
                self.population[i].fitness = self.fitness_function_normed(self.population[i].get_image(), self.target)
            
        if self.show_output:
            pbar = trange(len(self.population))
        else:
            pbar = range(len(self.population))
        # for i in pbar:
        #     if self.show_output:
        #         pbar.set_description_str("Simulating gen " + str(self.gen) + ": ")
        #     self.population[i].wait_for_simulation()
            
        for i, g in enumerate(self.population):
            self.population[i].update_with_fitness(g.fitness, count_members_of_species(self.population, self.population[i].species_id))
        
        if(self.config.novelty_selection_ratio_within_species > 0 or self.config.novelty_adjusted_fitness_proportion > 0):
            # novelties = novelty_ae.get_ae_novelties(self.population)
            novelties = np.zeros(len(self.population))
            for i, n in enumerate(novelties):
                self.population[i].novelty = n
                self.novelty_archive = update_solution_archive(self.novelty_archive, self.population[i], self.config.novelty_archive_len, self.config.novelty_k)
        
        for i in range(len(self.population)):
            if self.config.novelty_adjusted_fitness_proportion > 0:
                global_novelty = np.mean([g.novelty for g in self.novelty_archive])
                if(global_novelty==0): global_novelty=0.001
                adj_fit = self.population[i].adjusted_fitness
                adj_novelty =  self.population[i].novelty / global_novelty
                prop = self.config.novelty_adjusted_fitness_proportion
                self.population[i].adjusted_fitness = (1-prop) * adj_fit  + prop * adj_novelty 
        self.update_num_species_offspring()
    
    def update_num_species_offspring(self):
        for sp in self.all_species:
            sp.members = get_members_of_species(self.population, sp.id)
            sp.population_count = len(sp.members)
            if(sp.population_count <= 0): continue
            sp.avg_fitness = torch.mean(torch.stack([i.fitness for i in sp.members])).item()
            sp.sum_adj_fitness = torch.sum(torch.stack([i.adjusted_fitness for i in sp.members]))
        
        # print([sp.sum_adj_fitness for sp in self.all_species])
        # global_adj_fitness = torch.sum(torch.stack([i.adjusted_fitness for i in self.population]))
        # global_adj_fitness = torch.mean(torch.stack([i.adjusted_fitness for i in self.population]))
        fs = torch.stack([sp.sum_adj_fitness for sp in self.all_species if sp.population_count > 0])
        global_adj_fitness = torch.sum(fs)

        min_adj_fitness = torch.min(torch.stack([sp.sum_adj_fitness for sp in self.all_species if sp.population_count > 0]))
        max_adj_fitness = torch.max(torch.stack([sp.sum_adj_fitness for sp in self.all_species if sp.population_count > 0]))


        # update species, calculate allowed offspring
        for sp in self.all_species:
            if(sp.population_count<=0): sp.allowed_offspring = 0; continue
            sp.update(global_adj_fitness, sp.members, self.gen, self.config.species_stagnation_threshold, self.config.population_size, min_adj_fitness, max_adj_fitness)

        normalize_species_offspring(self.all_species, self.config)


        total_allowed_offspring = 0
        for sp in self.all_species:
            total_allowed_offspring += sp.allowed_offspring
        if total_allowed_offspring == 0:
            # total extinction
            # TODO
            sorted_species = sorted(self.all_species, key=lambda x: x.avg_fitness, reverse=True)
            sorted_species[0].allowed_offspring = self.config.population_size

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
        
    def print_fitnesses(self):
        div = calculate_diversity_full(self.population, self.all_species)
        num_species = count_number_of_species(self.population)
        print("Generation", self.gen, "="*100)
        print(f" |-Best: {self.get_best().id} ({self.get_best().fitness:.4f})")
        print(f" |  Average fitness: {torch.mean(torch.stack([i.fitness for i in self.population])):.7f} | adjusted: {torch.mean(torch.stack([i.adjusted_fitness for i in self.population])):.7f}")
        print(f" |  Diversity: std: {div[0]:.3f} | avg: {div[1]:.3f} | max: {div[2]:.3f}")
        print(f" |  Connections: avg. {get_avg_number_of_connections(self.population):.2f} max. {get_max_number_of_connections(self.population)}  | H. Nodes: avg. {get_avg_number_of_hidden_nodes(self.population):.2f} max: {get_max_number_of_hidden_nodes(self.population)}")
        for individual in self.population:
            print(f" |     Individual {individual.id} ({len(individual.hidden_nodes())}n, {len(list(individual.enabled_connections()))}c, s: {individual.species_id} fit: {individual.fitness:.4f}")
        
        print(" |-Species:")
        thresh_symbol = '='
        if self.config.num_generations>1 and self.species_threshold_over_time[self.gen-2]<self.species_threshold and self.species_threshold_over_time[self.gen-2]!=0:
            thresh_symbol = '▲' 
        if self.config.num_generations>1 and self.species_threshold_over_time[self.gen-2]>self.species_threshold:
            thresh_symbol = '▼'
        print(f" |  Count: {num_species} / {self.config.species_target} | threshold: {self.species_threshold:.2f} {thresh_symbol}") 
        print(f" |  Best species (avg. fitness): {sorted(self.all_species, key=lambda x: x.avg_fitness if x.population_count > 0 else -1000000000, reverse=True)[0].id}")
        for species in self.all_species:
            if species.population_count > 0:
                print(f" |    Species {species.id:03d} |> fit: {species.avg_fitness:.4f} | adj: {species.sum_adj_fitness:.4f} | stag: {self.gen-species.last_improvement} | pop: {species.population_count} | offspring: {species.allowed_offspring if species.allowed_offspring > 0 else 'X'}")

        print(f" Gen "+ str(self.gen), f"fitness: {self.get_best().fitness:.4f}")
        print()

    def neat_selection_and_reproduction(self):
        new_children = []
        # global_adj_fitness = torch.sum(torch.stack([i.adjusted_fitness for i in self.population]))
        # for sp in self.all_species:
        #     sp.members = get_members_of_species(self.population, sp.id)
        #     sp.population_count = len(sp.members)
        #     if(sp.population_count<=0): sp.allowed_offspring = 0; continue
        #     sp.update(global_adj_fitness, sp.members, self.gen, self.config.species_stagnation_threshold, self.config.population_size)
        # self.update_num_species_offspring()
        for sp in self.all_species:
            if(sp.population_count<=0): continue
            sp.current_champ = sp.members[0] # still sorted from before
            if(self.config.within_species_elitism > 0  and sp.allowed_offspring > 0):
                n_elites = min(sp.population_count, self.config.within_species_elitism, sp.allowed_offspring) 
                for i in range(n_elites):
                    # Elitism: add the elite and make one less offspring
                    new_child = copy.deepcopy(sp.members[i])
                    new_child.id = self.genome_type.get_id()
                    new_children.append(new_child)
                    sp.allowed_offspring-=1
                    
            if(len(sp.members)>1):
                new_members = []
                fitness_selected = round((1-self.config.novelty_selection_ratio_within_species) * self.config.species_selection_ratio * len(sp.members)) 
                if self.debug_output:
                    print("Species", sp.id, "fitness selected:", fitness_selected)
                new_members = sp.members[:fitness_selected] # truncation selection
                if (self.config.novelty_selection_ratio_within_species > 0):
                    novelty_members = sorted(sp.members, key=lambda x: x.novelty, reverse=True) 
                    novelty_selected = round(self.config.novelty_selection_ratio_within_species * self.config.species_selection_ratio * len(sp.members)) 
                    new_members.extend(novelty_members[:novelty_selected+1]) # truncation selection
                
                sp.members = new_members
                # members = tournament_selection(members, c, True, override_no_elitism=True) # tournament selection
            if (len(sp.members)==0):
                continue # no members in species
            # print("Creating", sp.allowed_offspring, "offspring for species", sp.id, "with", len(sp.members), "members")
            for i in range(sp.allowed_offspring):
                if len(sp.members) == 0: break
                # inheritance
                parent1 = np.random.choice(sp.members, size=max(len(sp.members), 1))[0] # pick 1 random parent
                #crossover
                if(self.config.do_crossover and parent1 and torch.rand(1)[0] < self.config.crossover_ratio):
                    other_id = -1
                    other_members = []
                    if(torch.rand(1)[0]<self.config.crossover_between_species_probability): # cross-species crossover (.001 in s/m07)
                        for sp2 in self.all_species:
                            if count_members_of_species(self.population, sp2.id) > 0 and sp2.id!=sp.id:
                                other_id = sp2.id
                        if(other_id>-1): 
                            other_members = get_members_of_species(self.population, other_id)
                    
                    parent2 = np.random.choice(other_members if other_id>-1 else sp.members, size=max(len(other_members), 1))[0] # allow parents to crossover with themselves
                    
                    if parent2:
                        child = parent1.crossover(parent2)
                else:
                    if parent1:
                        child = copy.deepcopy(parent1)
                        child.set_id(self.genome_type.get_id())
                    else:
                        continue

                self.mutate(child)
                assert child is not None
                new_children.extend([child]) # add children to the new_children list
        for sp in self.all_species:
            sp.members = [] # reset members for next generation
        return new_children

    def evolve(self, run_number = 1, show_output=True):
        self.start_time = time.time()
        try:
            self.run_number = run_number
            self.show_output = show_output or self.debug_output
            for i in range(self.config.population_size): # only create parents for initialization (the mu in mu+lambda)
                self.population.append(self.genome_type(self.config)) # generate new random individuals as parents
            
            if self.config.use_speciation:
                assign_species(self.all_species, self.population, self.population, self.species_threshold, Species)

            # Run NEAT
            pbar = trange(self.config.num_generations, desc=f"Run {self.run_number}")
            self.update_fitnesses_and_novelty()
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True) # sort by fitness
            self.solution = self.population[0]
            
            for self.gen in pbar:
                self.run_one_generation()
                pbar.set_postfix_str(f"f: {self.get_best().fitness:.4f} d:{self.diversity_over_time[self.gen-1]:.4f} s:{self.num_species}, t:{self.species_threshold:.3f}")
        except KeyboardInterrupt:
            raise KeyboardInterrupt()  
        self.end_time = time.time()     
        self.time_elapsed = self.end_time - self.start_time     

    def run_one_generation(self):
        if self.show_output:
            self.print_fitnesses()
        self.update_fitness_function()
        #-----------#
        # selection #
        #-----------#
        new_children = [] # keep children separate for now
        
        # elitism
        for i in range(self.config.population_elitism):
            new_children.append(copy.deepcopy(self.population[i])) # keep most fit individuals without mutating (should already be sorted)
        
        if(self.config.use_speciation):
            # i.e NEAT
            new_children.extend(self.neat_selection_and_reproduction()) # make children within species
            self.num_species = assign_species(self.all_species, self.population, new_children, self.species_threshold, Species) # assign new species ids
            self.population = new_children # replace parents with new children (mu, lambda)
            for c in self.population:
                assert c.species_id is not None
                
            # for sp in self.all_species:
                # print("Species", sp.id, "has", sp.population_count, "members")
            
        else:
            self.population = truncation_selection(self.population, self.config) # truncation
            new_children.extend(classic_selection_and_reproduction(self.config, self.population, self.all_species, self.gen, self.get_mutation_rates()))
            self.population += new_children # combine parents with new children (mu + lambda)
            self.num_species = 0
            
        #------------#
        # assessment #
        #------------#
        self.update_fitnesses_and_novelty() # evaluate CPPNs
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True) # sort by fitness
        self.solution = self.population[0]
        self.this_gen_best = self.solution
        
        #----------------#
        # record keeping #
        #----------------#
        # diversity:
        # std_distance, avg_distance, max_diff = calculate_diversity(self.population, self.all_species)
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

        # species
        # adjust the species threshold to get closer to the right number of species
        if self.config.use_speciation:
            delta = self.config.species_threshold_delta
            
            # TODO testing:
            # automatically determine delta
            # delta = avg_distance - self.diversity_over_time[self.gen-1]
            # delta *= 1.5
            # delta = abs(delta)
            # delta = max(self.config.species_threshold_delta, delta)
            ###############################
            
            if(self.num_species>self.config.species_target): self.species_threshold+=delta
            if(self.num_species<self.config.species_target): self.species_threshold-=delta
            self.species_threshold = max(0.010, self.species_threshold)
            # self.species_threshold = max(0.10, self.species_threshold)
        
        
        self.species_over_time[self.gen:] = self.num_species
        self.species_threshold_over_time[self.gen:] = self.species_threshold
        self.species_pops_over_time.append([s.population_count for s in sorted(self.all_species, key=lambda x: x.id)])
        
        champs = get_current_species_champs(self.population, self.all_species)
        self.species_champs_over_time.append(champs) 

        # if self.show_output:
            # self.save_best_network_image()
    
    
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


def classic_selection_and_reproduction(c, population, all_species, generation_num, mutation_rates):
    new_children = []
    while len(new_children) < c.population_size:
        # inheritance
        parent1 = np.random.choice(population, size=1)[0] # pick 1 random parent

        #crossover
        if(c.do_crossover):
            if c.use_speciation:
                parent2 = np.random.choice(get_members_of_species(population, parent1.species_id), size=1)[0] # note, we are allowing parents to crossover with themselves
            else:
                parent2 = np.random.choice(population, size=1)[0]
            child = parent1.crossover(parent2)
        else:
            child = copy.deepcopy(parent1)

        # mutation
        child.mutate(mutation_rates)
        
        new_children.extend([child]) # add children to the new_children list

    return new_children

def truncation_selection(population, c):
    assert hasattr(c, "truncation_threshold"), "truncation_threshold not set in config"
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True) # sort population by fitness (from high to low)
    # print([i.fitness for i in sorted_population])
    return sorted_population[:int(len(sorted_population)*c.truncation_threshold)] # truncation

def tournament_selection(population, c, use_adjusted_fitness=False, override_no_elitism=False):
    new_population = []
    if(not override_no_elitism): 
        for i in range(c.population_elitism):
            new_population.append(population[i]) # keep best genomes (elitism)

    # fitness
    while len(new_population) < (1-c.novelty_selection_ratio_within_species)*c.population_size:
        tournament = np.random.choice(population, size = min(c.tournament_size, len(population)), replace=False)
        if(use_adjusted_fitness):
            tournament = sorted(tournament, key=lambda genome: genome.adjusted_fitness, reverse=True)
        else:
            tournament = sorted(tournament, key=lambda genome: genome.fitness, reverse=True)
        new_population.extend(tournament[:c.tournament_winners])  

    # novelty
    if(c.novelty_selection_ratio_within_species > 0):
        sorted_pop = sorted(population, key=lambda genome: genome.novelty, reverse=True) # sort the full population by each genome's fitness (from highers to lowest)
        while len(new_population) < c.num_parents:
            tournament = np.random.choice(sorted_pop, size = min(c.tournament_size, len(sorted_pop)), replace=False)
            tournament = sorted(tournament, key=lambda genome: genome.novelty, reverse=True)
            new_population.extend(tournament[:c.tournament_winners])  
    
    return new_population  


def calculate_diversity(population, all_species):
    # Compares 1 representative from every species against each other
    reps = []
    for species in all_species:
        members = get_members_of_species(population, species.id)
        if(len(members)<1): continue
        reps.append(np.random.choice(members, 1)[0])
    diffs = []
    for i in reps:
        for j in reps:
            if i== j: continue
            diffs.append(i.genetic_difference(j))

    std_distance = np.std(diffs)
    avg_distance = np.mean(diffs)
    max_diff = np.max(diffs)if(len(diffs)>0) else 0
    return std_distance, avg_distance, max_diff

    
def calculate_diversity_full(population, all_species=None):
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

def update_solution_archive(solution_archive, genome, max_archive_length, novelty_k):
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

