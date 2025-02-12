from itertools import islice
import random
import copy
from collections.abc import Sequence

class Pool(Sequence):

    def __init__(self):
       self.population = []

    def dict_to_string(self, key_dict):
        r"""
        dict: {dropped mod id: percent}
        Example:
            {1:30%, 2:70%} -> "1:0.3,2:0.7"
        """
        assert(type(key_dict) is dict)
        # sort key, this is necessary because we will the key to identify an 
        # individual in a pool, if random order is allowed this will cause
        # duplication, e.g. (1:0.2, 2:0.8) != (2:0.8, 1:0.2)

        # order dict
        # key_dict = OrderedDict(sorted(key_dict.items()))
        key_dict = dict(sorted(key_dict.items()))
        
        # converet OrderedDict to string
        return str(key_dict) 


    def size(self):
        return len(self.population)

    def add(self, individual):
        self.population.append(individual)

    def copy_add(self, individual):
        individual = copy.deepcopy(individual)
        self.population.append(individual)

    def pop(self, index=None):
        if index is None:
            self.population.pop()
        else:
            self.population.pop(index)

    def __getitem__(self, key):
        return self.population[key]

    def __setitem__(self, key, newvalue):
        self.population[key] = newvalue

    def __len__(self):
        return len(self.population)

    # def get(self, key):
    #     if type(key) is dict:
    #         key = self.dict_to_string(key)
    #     for pair in self.population:
    #         if key == pair[0]:
    #             return pair[1]
    #     raise RuntimeError("Can't find the key in the pool")

    def order(self, key=None):
        # x[1] is the Individual object
        if key == "memory":
            self.population = sorted(self.population, key=lambda x: x.memory, reverse=False)
        elif key == "latency":
            self.population = sorted(self.population, key=lambda x: x.latency, reverse=False)
        else:
            raise NotImplemented("order pool by other metric not supported yet")

    def select_top(self, top_k, key="memory"):
        # return list(islice(self.population.items(), top_k))
        self.order(key=key)
        return self.population[:top_k]

    def random_elem(self):
        return random.choice(self.population) 

    def shuffle(self, exclude=0):
        def shuffle_slice(a, start, stop):
            i = start
            while i < stop-1:
                idx = random.randrange(i, stop)
                a[i], a[idx] = a[idx], a[i]
                i += 1
        return shuffle_slice(self.population, exclude, len(self.population))

    def select(self, start=None, stop=None):
        return self.population[start:stop]


    def print(self):
        print("===== Population =====")
        for elem in self.population:
            elem.print()


