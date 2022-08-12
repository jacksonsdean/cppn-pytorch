import copy
import math
import random
import traceback
import numpy as np
import torch

class Species:
    def __init__(self, _id) -> None:
        self.id = _id
        self.sum_adj_fitness = torch.tensor(0.0, dtype=torch.float32)
        self.avg_fitness = torch.tensor(0.0, dtype=torch.float32)
        self.allowed_offspring = 0
        self.population_count = 0
        self.last_fitness =  self.sum_adj_fitness
        self.last_improvement = 0
        self.current_champ = None
    
    def update(self, global_adjusted_fitness, members, gen, stagnation_threshold, total_pop, min_adj_fitness, max_adjusted_fitness):
        self.sum_adj_fitness = torch.sum(torch.stack([i.adjusted_fitness for i in members]))
        # self.mean_adj_fitness = torch.mean(torch.stack([i.adjusted_fitness for i in members]))
        
        # self.sum_adj_fitness = torch.mean(torch.stack([i.adjusted_fitness for i in members]))
        # self.avg_fitness =  torch.mean(torch.stack([i.fitness for i in members]))
        
        
        self.members= members
        self.population_count = len(members)
        
        if(self.avg_fitness > self.last_fitness):
            self.last_improvement = gen
            self.last_fitness = self.avg_fitness

        # Every species is assigned a potentially different number of offspring in proportion to the sum of ad-
        # justed fitnesses of its member organisms. Species then reproduce by first eliminating
        # the lowest performing members from the population. The entire population is then
        # replaced by the offspring of the remaining organisms in each species.

        if(gen - self.last_improvement >= stagnation_threshold):
            # stagnation
            self.allowed_offspring = 0
            for member in members:
                member.species_id = None
        else:
            try:
                # nk = (Fk/Ftot)*P 
                fit_range = max_adjusted_fitness - min_adj_fitness
                if fit_range == 0:
                    self.allowed_offspring = self.population_count # TODO
                else:
                    
                    # proportion = self.sum_adj_fitness / global_adjusted_fitness
                    proportion = (self.sum_adj_fitness -min_adj_fitness) / fit_range
                    
                    # print("global_adjusted_fitness: ", global_adjusted_fitness, "self.sum_adj_fitness: ", self.sum_adj_fitness, "proportion: ", proportion)
                    self.allowed_offspring = int(torch.round(total_pop * (proportion)))
                    
                    # print(self.id, global_adjusted_fitness, self.sum_adj_fitness, proportion, self.allowed_offspring)
                    # self.allowed_offspring = int(torch.round(self.population_count * (self.sum_adj_fitness / global_adjusted_fitness)))
                    # self.allowed_offspring = int(torch.round(self.population_count * (self.sum_adj_fitness / global_adjusted_fitness)))
                if self.allowed_offspring < 0: self.allowed_offspring = 0
            except ArithmeticError:
                print(f"error while calc allowed_offspring: pop:{self.population_count} fit:{self.sum_adj_fitness} glob: {global_adjusted_fitness}")
                traceback.print_exc()
            except ValueError:
                print(f"error while calc allowed_offspring: pop:{self.population_count} fit:{self.sum_adj_fitness} glob: {global_adjusted_fitness}")
                traceback.print_exc()


def get_adjusted_fitness_of_species(population, species_id:int):
    return torch.mean(torch.stack([i.adjusted_fitness for i in population if i.species_id == species_id]))
def get_fitness_of_species(population, species_id:int):
    return torch.mean(torch.stack([i.fitness for i in population if i.species_id == species_id]))
def count_members_of_species(population, species_id:int):
    return len(get_members_of_species(population, species_id))
def get_members_of_species(population, species_id:int):
    return [ind for ind in population if ind.species_id == species_id]
def count_number_of_species(population):
    species = []
    for ind in population:
        if(ind.species_id not in species):
            species.append(ind.species_id)
    return len(species)

def get_current_species_champs(population, all_species):
    for sp in all_species:
        members = get_members_of_species(population, sp.id)
        sp.population_count = len(members)
        if(sp.population_count==0): continue
        members= sorted(members, key=lambda x: x.fitness, reverse=True)
        sp.current_champ = members[0]
    return [sp.current_champ for sp in all_species if (sp.population_count>0 and sp.current_champ is not None)]


def assign_species(all_species, parents, children, threshold, SpeciesClass):
    reps = {}
    if len(all_species) == 0:
        # first time through, create species 0
        all_species.append(SpeciesClass(0))
        np.random.choice(children, 1)[0].species_id = 0
        
    for s in all_species:
        s.members = get_members_of_species(parents, s.id)
        s.population_count = len(s.members)
        if(s.population_count<1): continue
        reps[s.id] = np.random.choice(s.members, 1)[0]
        # print(f"species {s.id} has {s.population_count} members and rep is {reps[s.id].id}")
    
    # for r in reps:
    #     print(f"species {r} has {reps[r].id} as representative")
        
    # The Genome Loop:
    for g in children:
        # – Take next genome g from P
        placed = False

        # print("individual {}".format(g.id))
        possible_reps = list(reps.keys())
        if(g.species_id  in reps.keys() and g.species_comparision(reps[g.species_id], threshold)):
            # check if g is similar to it's parents species representative
            g.species_id = s.id
            placed = True
            
        else:
            g.species_id = None #find a new home
            
        # random.shuffle(possible_reps)
        # – The Species Loop:
        for s_index in possible_reps:
            if placed: break
            # print(f"\tcheck species {s_index}")
            s = all_species[s_index]
            if not s.id in reps.keys() or reps[s.id] is None:
                raise Exception(f"species {s.id} has no representative")
                continue
            # ·get next species s from S
            s.members = get_members_of_species(parents, s.id)
            # species_pop = s.members
            s.population_count = len(s.members)

            if(s.population_count<1): continue
            # print("\t gen diff:", g.genetic_difference(reps[s.id]), "thresh:", threshold, "same?", g.species_comparision(reps[s.id], threshold))
            if(g.species_comparision(reps[s.id], threshold)):
                # print("\t\tindividual {} is similar to representative {} of species {}".format(g.id, reps[s.id].id, s.id) )
                # ·If g is compatible with s, add g to s
                g.species_id = s.id
                placed = True
                break
        if(not placed):
            # ∗If all species in S have been checked, create new species and place g in it
            # print("no species for individual {}".format(g.id))
            new_id = len(all_species)
            all_species.append(SpeciesClass(new_id))
            all_species[-1].population_count = 1
            g.species_id = new_id
            reps[new_id] = g
        assert g.species_id is not None, f"g.species_id is None: {g.id}"
            # print(f"new species {new_id} created with rep", g.id)
    # for s in all_species:
        # print(f"species {s.id} has {s.population_count} members: {[i.id for i in get_members_of_species(population, s.id)]}")
    
    # TODO: slow
    for sp in all_species:
        sp.members = get_members_of_species(children, sp.id)
        sp.population_count = len(sp.members)
        # print("species {}, members: {}, pop: {}".format(sp.id, len(sp.members), sp.population_count))
    
    return count_number_of_species(children)

def normalize_species_offspring(all_species, c):
    # Normalize the number of allowed offspring per species so that the total is close to population_size
    total_offspring = np.sum([s.allowed_offspring for s in all_species])
    target_children = c.population_size
    target_children -= c.population_elitism # first children will be most fit from last gen 

    if(total_offspring == 0): total_offspring = 1 # TODO FIXME (total extinction)
    
    norm = c.population_size / total_offspring
    
    for sp in all_species:
        try:
            sp.allowed_offspring = int(np.round(sp.allowed_offspring * norm))
        except ValueError as e:
            print(f"unexpected value during species offspring normalization, ignoring: {e} offspring: {sp.allowed_offspring} norm:{norm}")
            continue

    return norm

def normalize_species_offspring_exact(all_species, population_size):
    # Jackson's method (always exact population_size)
    # if there are not enough offspring, assigns extras to top (multiple) species,
    # if there are too many, takes away from worst (multiple) species
    total_offspring = torch.sum(torch.stack([s.allowed_offspring for s in all_species]))
    adj = 1 if total_offspring<population_size else -1
    sorted_species = sorted(all_species, key=lambda x: x.sum_adj_fitness, reverse=(total_offspring<population_size))
    while(total_offspring!=population_size):
        for s in sorted_species:
            if(s.population_count == 0 or s.allowed_offspring == 0): continue
            s.allowed_offspring+=adj
            total_offspring+=adj
            print("adj=", adj)
            if(total_offspring==population_size): break
