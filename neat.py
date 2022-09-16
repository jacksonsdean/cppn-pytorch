import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from cppn_neat.cppn import *
from cppn_neat.evolutionary_algorithm import EvolutionaryAlgorithm, calculate_diversity_full
from cppn_neat.util import *
from cppn_neat.species import *
import copy 

import random
import copy
import os

class NEAT(EvolutionaryAlgorithm):
    def __init__(self, target, config, debug_output=False) -> None:
        super().__init__(target, config, debug_output)
        self.all_species = []
        self.species_threshold = self.config.init_species_threshold
        # normalize fitness function
        self.update_fitness_function()
    
   
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
                child = None # the child to be added to the population
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

                assert child is not None
                self.mutate(child)
                assert child is not None
                new_children.extend([child]) # add children to the new_children list
        for sp in self.all_species:
            sp.members = [] # reset members for next generation
        return new_children

    def evolve(self, run_number = 1, show_output=True):
        super().evolve()
        try:
            if self.config.use_speciation:
                assign_species(self.all_species, self.population, self.population, self.species_threshold, Species)
            # Run NEAT
            pbar = trange(self.config.num_generations, desc=f"Run {self.run_number}")
            self.update_fitnesses_and_novelty()
            for g in self.population:
                assert g.fitness >= 0, f"fitness must be non-negative for now in NEAT, but got {g.fitness}"

            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True) # sort by fitness
            self.solution = self.population[0]
            
            for self.gen in pbar:
                self.run_one_generation()
                pbar.set_postfix_str(f"f: {self.get_best().fitness:.4f} d:{self.diversity_over_time[self.gen-1]:.4f} s:{self.num_species}, t:{self.species_threshold:.3f}")
        except KeyboardInterrupt:
            self.end_time = time.time()     
            self.time_elapsed = self.end_time - self.start_time     
            raise KeyboardInterrupt()  

    def run_one_generation(self):
        super().run_one_generation()
       
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
        
        
        #----------------#
        # record keeping #
        #----------------#
        self.record_keeping()
       
    def record_keeping(self):
        super().record_keeping()
        # species
        # adjust the species threshold to get closer to the right number of species
        if self.config.use_speciation:
            delta = self.config.species_threshold_delta
            if(self.num_species>self.config.species_target): self.species_threshold+=delta
            if(self.num_species<self.config.species_target): self.species_threshold-=delta
            self.species_threshold = max(0.010, self.species_threshold)
            # self.species_threshold = max(0.10, self.species_threshold)
        
        self.species_over_time[self.gen:] = self.num_species
        self.species_threshold_over_time[self.gen:] = self.species_threshold
        self.species_pops_over_time.append([s.population_count for s in sorted(self.all_species, key=lambda x: x.id)])
        
        champs = get_current_species_champs(self.population, self.all_species)
        self.species_champs_over_time.append(champs) 
    
    def update_fitnesses_and_novelty(self):
        super().update_fitnesses_and_novelty()
        for i, g in enumerate(self.population):
            self.update_individual_with_fitness(g, g.fitness, count_members_of_species(self.population, g.species_id))
            
        for i in range(len(self.population)):
                if self.config.novelty_adjusted_fitness_proportion > 0:
                    global_novelty = np.mean([g.novelty for g in self.novelty_archive])
                    if(global_novelty==0): global_novelty=0.001
                    adj_fit = self.population[i].adjusted_fitness
                    adj_novelty =  self.population[i].novelty / global_novelty
                    prop = self.config.novelty_adjusted_fitness_proportion
                    self.population[i].adjusted_fitness = (1-prop) * adj_fit  + prop * adj_novelty 
        self.update_num_species_offspring()
    
    def update_individual_with_fitness(self, individual, fit, num_in_species):
        assert fit >= 0, f"fitness must be non-negative for now, but got {fit}"
        individual.fitness = fit
        if(num_in_species > 0):
            individual.adjusted_fitness = (individual.fitness / num_in_species)  # local competition
            assert not torch.isnan(individual.adjusted_fitness).any(), f"adjusted fitness was nan: fit: {individual.fitness} n_in_species: {num_in_species}"
        else:
            self.adjusted_fitness = individual.fitness
            raise(Exception("num_in_species was 0"))
        
    def print_fitnesses(self):
        super().print_fitnesses()
        print(" |-Species:")
        num_species = count_number_of_species(self.population)
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

    
