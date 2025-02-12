import copy
import random
import os
import json

from dropinf.optimizer.identifier import Identifier
from dropinf.optimizer.individual import Individual
from dropinf.optimizer.pool import Pool
from dropinf.utils.model import copy_model_to_gpus
from dropinf.utils.cuda import synchronize_gpus, all_max_memories, dict_to
from dropinf.profiler.utils import load_profile_result

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

class Optimizer:
   
    def __init__(
            self,
            accuracy_objs=None,
            profiler_results_filename=None,
            batch_size=None,
            infer_func=None,
            input_manager=None,
            results_dir=None,
            experiment=None):
        
        """
        dropped_mods contains all possible modalities that can be dropped
        Example:
            given a model with 2 modality: audio, video
            dropped_mods = audio, video, None
        """

        assert(accuracy_objs is not None)
        self.accuracy_objs = accuracy_objs
        self.profiler_results_filename = profiler_results_filename
        self.batch_size = batch_size
        
        self.infer = infer_func
        self.input_manager = input_manager

        self.results_dir = results_dir
        self.experiment = experiment

        self.mod_accuracy = {} # the avg accuracy for each dropping policy

        self.candidates_pools = self.init_candidates_pools(accuracy_objs=
                self.accuracy_objs)

        # TODO:
        self.timer_start = torch.cuda.Event(enable_timing=True)
        self.timer_end = torch.cuda.Event(enable_timing=True)


    def init_candidates_pools(self, accuracy_objs=None):
        """
        For each accuracy objective, it has a pool of dropping policies that
        meets the acc obj.

        Args:
            accuracy_objs: [acc]
        """
        accuracy_objs = sorted(accuracy_objs)
        candidates_pools = {}
        for accuracy_obj in accuracy_objs:
            candidates_pools[accuracy_obj] = Pool()

        return candidates_pools


    def init_population(
            self,
            profile_data=None,
            population_size=10,
            objective=None):
        
        """
        
        Args:
            profile_data: json file read
            batch_size: profile_data can contain mutliple batch sizes, pick one
                    to proceed
        """
        
        dropped_mods = profile_data.keys()
        self.identifier = Identifier(dropped_mods)

        self.pool = Pool()
        self.basic_pool = Pool() # this pool only contains basic mod combos

        loop_id = 0
        for dropped_mod, batch_metric in profile_data.items():
            metric = batch_metric[str(self.batch_size)]
            avg_acc = metric["accuracy"][0]
            avg_max_mem = metric["memory"][0]
            avg_lat = metric["latency"][0]

            self.mod_accuracy[dropped_mod] = avg_acc

            policy = {dropped_mod: self.batch_size}
            individual = Individual(identity=self.create_id(policy=policy),
                    policy=policy,
                    latency=avg_lat,
                    memory=avg_max_mem, avg_acc=avg_acc)
            self.basic_pool.add(individual)

            # if individual.avg_acc >= self.accuracy_obj and \
            if loop_id < population_size:
                policy = {dropped_mod: self.batch_size}
                elem = Individual(identity=self.create_id(policy=policy),
                        policy=policy,latency=avg_lat,
                    memory=avg_max_mem, avg_acc=avg_acc)
                self.pool.add(elem)
            loop_id += 1

        # fill up remaining space in pool if it's not filled up

        self.fill_pool(pool=self.pool, size=population_size)
        self.basic_pool.order(key=objective)
        self.pool.order(key=objective)

        
    # def prune(self):
    #     r"""
    #     Remove individuals whose avg_accuracy is below target accuracy objective
    #     """
    #     self.pool.population = [individual for individual in self.pool \
    #             if individual.avg_acc >= self.accuracy_obj]

    def add_candidates(self, source=None, to=None):
        """

        Args:
            source: self.pool
            to: candidates_pools
        """
        for individual in source:
            if individual.avg_acc is None or individual.memory is None or \
                    individual.latency is None:
                raise RuntimeError("individual not scored")
            for candidate_acc, candidate_pool in to.items():
                if individual.avg_acc >= candidate_acc:
                    candidate_pool.copy_add(individual)

    def prune_candidates_pools(
            self,
            candidates_pools=None,
            key=None,
            pool_size=None):

        for candidate_acc, candidate_pool in candidates_pools.items():
            candidate_pool.order(key=key)
            # remove extra
            while len(candidate_pool) > pool_size:
                candidate_pool.pop()

    def merge_pools(self, candidates_pools=None, pool_size=None):
        new_pool = Pool()
        counter = 0
        for i in range(pool_size):
            for candidate_acc, candidate_pool in candidates_pools.items():
                if i < len(candidate_pool):
                    new_pool.copy_add(candidate_pool[i])
                    counter += 1
                    if counter >= pool_size:
                        return new_pool

        return new_pool


    def warmup(
            self,
            models=None,
            dataloader=None,
            devices=None,
            warmup_steps=5):

        num_gpus = len(devices)
        warmup_steps *= num_gpus
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                device_index = i_batch % num_gpus
                device = devices[device_index]
                sample_batched = dict_to(sample_batched, device=device)
                _ = self.infer(models[device_index], sample_batched)
                if i_batch >= warmup_steps - 1:
                    break

    def optimize(
            self,
            model=None,
            dataloader=None,
            loop=6,
            population_size=10,
            elite_count=1,
            crossover_fraction=0.7,
            objective="memory",
            devices=None,
            warmup_steps=2):

        # load profiler data
        pd = load_profile_result(result_path=self.profiler_results_filename)
        self.init_population(profile_data=pd,
                population_size=population_size,
                objective=objective)
        
        model.eval()

        models = copy_model_to_gpus(model=model, devices=devices)
        synchronize_gpus(devices=devices)

        self.warmup(models=models, dataloader=dataloader, devices=devices,
                warmup_steps=warmup_steps)
        synchronize_gpus(devices=devices)

        # TODO more terminating conditions
        for i in range(loop):
            # self.prune()
            self.score(pool=self.pool, dataloader=dataloader,
                    models=models, sample_iter=3, trials=2,
                    devices=devices)

            self.add_candidates(source=self.pool, to=self.candidates_pools)
            # sort each candidate pool based on objective, then limit its size
            self.prune_candidates_pools(candidates_pools=self.candidates_pools,
                    key=objective, pool_size=population_size)

            self.dump_candidates(self.candidates_pools,
                    dump_dir=self.results_dir,
                    filename=self.experiment + "_" + str(i) + ".json")

            # merge the best performer from all pools
            self.pool = self.merge_pools(candidates_pools=self.candidates_pools,
                    pool_size=population_size)
            self.pool.order(key=objective)

            self.fill_pool(pool=self.pool, size=population_size)

            self.pool.order(key=objective)

            self.crossover(pool=self.pool,
                    basic_pool=self.basic_pool,
                    population_size=population_size,
                    elite_count=elite_count,
                    crossover_fraction=crossover_fraction)

            self.mutation(pool=self.pool,
                    population_size=population_size,
                    elite_count=elite_count,
                    crossover_fraction=crossover_fraction)

            self.pool.print()

        print("----- end -----")
        self.print_candidates(candidates_pools=self.candidates_pools, key=objective)
        self.pool.print()


    def create_id(self, policy=None):
        identity = {}
        for k, v in policy.items():
            mod_id = self.identifier.get_mod_id(mod_name=k)
            identity[mod_id] = v

        # sort the dict
        return str(dict(sorted(identity.items())))


    def fill_pool(self, pool=None, size=0):
        current_size = pool.size()
        remaining_size = size - current_size 
        for i in range(remaining_size):
            # pool_individual = copy.deepcopy(pool.random_elem())
            random_index = random.randint(0, current_size-1)
            pool_individual = copy.deepcopy(pool[random_index])
            pool.add(pool_individual)


    def order_fitness(self, pool=None, key="memory"):
        """
        order the memory pool by the given objective

        .. example::
            {"audio:1.0" : 800 mb, "video:1.0" : 600 mb} ->
                {"video:1.0" : 600 mb, "audio:1.0" : 800 mb}
        """
        pool.order(key=key)
        
         
    def select_top(self, pool=None, top_k=2):
        if top_k >= pool.size():
            top_k = pool.size()

        return pool.select_top(top_k)
        
    def crossover(
            self,
            population_size=None,
            elite_count=None,
            crossover_fraction=None,
            pool=None,
            basic_pool=None):
        """
        encourage mutation, discourage crossover
        """
        # 1 population : order by mem, but under acc constraint
        # 2 population: order by mem only
        
        # shuffel pool besides top_k, so we can randomly choose parents to corssover
        pool.shuffle(exclude=elite_count)

        # choose the crossover from pool
        crossover_num = int((population_size - elite_count * crossover_fraction) * crossover_fraction)
        parents = pool[elite_count:elite_count+crossover_num]
        for individual in parents:
            self.crossover_individual(individual, basic_pool)

    def crossover_individual(self, individual, basic_pool):
        # randomly pick from basic pool
        mate = basic_pool.random_elem()

        # check if they are the same
        if individual.identity == mate.identity:
            # mutate if more than one mod in policy
            self.mutate_elem(individual)
        else:
            # combine policy
            # 1 identitical drop mod, different percent
            if individual.share_modality(mate):
                self.mutate_elem(individual)
            # 2 not identitcal drop mod, have overlap
            # 3 not identitcal drop mod, have no overlap
            else:
                # bring in new combo from basic pool
                # First, decide a scale down factor for me
                scale_down_factor = random.randint(1, 20) / float(20)
                    
                # scale down current policy
                new_policy = {k : int(v*scale_down_factor) for (k, v) in individual.policy.items()}
                scaled_down_size = sum(new_policy.values())
                # if the size of broughtt-in mod is 0, don't update
                if scaled_down_size >= self.batch_size:
                    self.mutate_elem(individual)
                else:
                    # add new policy
                    new_mod_batch_size = self.batch_size - scaled_down_size
                    new_mod, _ = next(iter(mate.policy.items()))
                    new_policy[new_mod] = new_mod_batch_size
                    # update individual new policy, id, invalidate latency, mem, calc avg_acc
                    # TODO, might be able to estimate the lat and mm
                    self.update_policy(individual, new_policy)


    def update_policy(self, individual, new_policy):
        individual.update_policy(new_policy)
        new_id = self.create_id(policy=new_policy)
        individual.identity = new_id

        individual.latency = None
        individual.memory = None
        individual.throughput = None
        individual.clear_internal_states()

        individual.avg_acc = 0.0
        for k, v in individual.policy.items():
            ratio = v / float(self.batch_size)
            individual.avg_acc += ratio * self.mod_accuracy[k]


    def mutation(
            self, 
            pool=None,
            population_size=None,
            elite_count=None,
            crossover_fraction=None):

        # crossover should already shuffled
        crossover_num = int((population_size - \
                elite_count * crossover_fraction) * crossover_fraction)
        parents = pool[elite_count+crossover_num:]
        for individual in parents:
            self.mutate_elem(individual)


    def mutate_elem(self, individual):
        def constrained_sum_sample_pos(n, total):
            """Return a randomly chosen list of n positive integers summing to total.
            Each such list is equally likely to occur."""

            dividers = sorted(random.sample(range(1, total), n - 1))
            return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
        # if there is only one policy
        if len(individual.policy) == 1:
            return
        # if there is more than one policy
        elif len(individual.policy) > 1:
            new_policy_batch_sizes = constrained_sum_sample_pos(len(individual.policy), self.batch_size)
            assert(len(new_policy_batch_sizes) == len(individual.policy))
            new_policy = {k : new_policy_batch_sizes[i]  for i, (k, _) in \
                    enumerate(individual.policy.items())}
            self.update_policy(individual, new_policy)
        else:
            raise RuntimeError("policy can't be empty")
    
    def score_helper(
            self,
            individual=None,
            dataloader=None,
            model=None, 
            objective="memory",
            sample_iter=6,
            device=None):

        with torch.no_grad():

            timer_start = torch.cuda.Event(enable_timing=True)
            timer_end = torch.cuda.Event(enable_timing=True)
            
            total_max_mem = 0
            total_lat = 0

            for i_batch, sample_batched in enumerate(dataloader):
               
                #split_batch according to individual[policy]
                splitted = self.input_manager.split_batch(sample_batched,
                        split_sections=list(individual.policy.values()))
                # drop_data in batch
                for i, (drop_mod, _) in enumerate(individual.policy.items()):
                    # turn drop_mod into a list, drop_mod is a string seperated with .
                    splitted[i] = self.input_manager.remove_mod(splitted[i],
                            removed_modality=drop_mod.split(","))
                    splitted[i] = self.input_manager.cuda(splitted[i], device=device)
        
                torch.cuda.reset_max_memory_allocated(device=device)
                # measurement()
                timer_start.record()
                for sub_batch in splitted:
                    _ = self.infer(model, sub_batch)

                # measurement()
                timer_end.record()
                torch.cuda.synchronize(device=device)
                lat = timer_start.elapsed_time(timer_end)
                max_mem = torch.cuda.max_memory_allocated(device=device)
                total_max_mem += max_mem
                total_lat += lat

                if i_batch + 1 >= sample_iter:
                    break
            
            individual.memory = total_max_mem / sample_iter
            individual.latency = total_lat / sample_iter


    def num_unevaluted(self, pool=None):
        counter = 0
        for individual in pool:
            if not individual.is_evaluated():
                counter += 1
        return counter
    
    def score(
            self,
            pool=None,
            dataloader=None,
            models=None,
            sample_iter=None,
            trials=3,
            devices=None):
        # processes = []
        # for i_individual, individual in enumerate(self.pool):
        #     if individual.is_evaluated():
        #         continue
        #     # assign the individual to a process
        #     device = i_individual % num_gpus
        #     keywords = {"individual": individual, "dataloader": dataloader,
        #             "model": models[device], "objective": objective,
        #             "sample_iter": sample_iter, "device": device}
        #     proc = mp.Process(target=self.score_helper, kwargs=keywords)
        #     processes.append(proc)
        #     if len(processes) == num_gpus or i_individual >= len(self.pool) - 1:
        #         for p in processes:
        #             p.start()
        #         for p in processes:
        #             p.join()
        #         synchronize_gpus(num_gpus=num_gpus)
        #         processes = []


            # continue


#        with torch.no_grad():
#            for individual in self.pool:
#                if individual.is_evaluated():
#                    continue
#                total_max_mem = 0
#                total_lat = 0
#                for i_batch, sample_batched in enumerate(dataloader):
#                    if i_batch > sample_iter - 1:
#                        break


#                    #split_batch according to individual[policy]
#                    splitted = self.input_manager.split_batch(sample_batched,
#                            split_sections=list(individual.policy.values()))
#                    # drop_data in batch
#                    for i, (drop_mod, _) in enumerate(individual.policy.items()):
#                        # turn drop_mod into a list, drop_mod is a string seperated with .
#                        splitted[i] = self.input_manager.remove_mod(splitted[i], removed_modality=drop_mod.split(","))
            
                    
#                    for i in range(trials):
#                        torch.cuda.reset_max_memory_allocated()
#                        # measurement()
#                        self.timer_start.record()
#                        for sub_batch in splitted:
#                            _ = self.infer(model, sub_batch)

#                        # measurement()
#                        self.timer_end.record()
#                        torch.cuda.synchronize()
#                        lat = self.timer_start.elapsed_time(self.timer_end)
#                        max_mem = torch.cuda.max_memory_allocated()
#                        total_max_mem += max_mem
#                        total_lat += lat
                
#                individual.memory = total_max_mem / (sample_iter * trials)
#                individual.latency = total_lat / (sample_iter * trials)


        # create recorders
        
        num_gpus = len(devices)
        timers_start = [None] * num_gpus
        timers_end = [None] * num_gpus
        for i in range(num_gpus):
            timers_start[i] = torch.cuda.Event(enable_timing=True)
            timers_end[i] = torch.cuda.Event(enable_timing=True)


        num_unevaluated = self.num_unevaluted(pool=self.pool)


        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                if i_batch > sample_iter - 1:
                    break

                num_uneval = num_unevaluated
                individual_recorded = []
                id_individual = 0
                for individual_counter, individual in enumerate(self.pool):
                    if individual.is_evaluated():
                        continue
                    
                    num_uneval -= 1

                    individual_recorded.append(individual)
                    device_index = id_individual % num_gpus
                    device = devices[device_index]

                    copied_batch = copy.deepcopy(sample_batched)

                    splitted = self.input_manager.split_batch(copied_batch,
                            split_sections=list(individual.policy.values()))

                    for i, (dropped_mods, _) in enumerate(individual.policy.items()):
                        splitted[i] = self.input_manager.remove_mod(splitted[i],
                                removed_modality=dropped_mods.split(","))
                        splitted[i] = self.input_manager.cuda(splitted[i],
                                device=device)

                    # perform infer if all gpus are copied
                    torch.cuda.reset_max_memory_allocated(device=device)
                    timer_start = timers_start[device_index]
                    timer_end = timers_end[device_index]
                    device_stream = torch.cuda.current_stream(device=device)

                    timer_start.record(stream=device_stream)
                    for sub_batch in splitted:
                        _ = self.infer(models[device_index], sub_batch)
                    timer_end.record(stream=device_stream)

                    if device_index >= num_gpus - 1 or individual_counter >= \
                            len(self.pool) - 1 or num_uneval <= 0:
                        synchronize_gpus(devices=devices)
                        
                        max_memories = all_max_memories(devices=devices)
                        for i, ind in enumerate(individual_recorded):
                            t_s = timers_start[i]
                            t_e = timers_end[i]
                            ind._total_latency += t_s.elapsed_time(t_e)
                            ind._total_max_memory += max_memories[i]

                        individual_recorded = []

                    id_individual += 1
                
            # average memory and latency
            synchronize_gpus(devices=devices)
            for individual in self.pool:
                if not individual.is_evaluated():
                    individual.memory = individual._total_max_memory / \
                            sample_iter
                    individual.latency = individual._total_latency / \
                            sample_iter
                    individual.throughput = sample_iter * self.batch_size / \
                            (individual._total_latency / 1000)
                    individual.clear_internal_states()

    def print_candidates(self, candidates_pools=None, key="memory"):
        for candidate_acc, candidate_pool in candidates_pools.items():
            print("===== candidate accuracy: " + str(candidate_acc) + " =====")
            candidate_pool.order(key=key)
            candidate_pool.print()

    def dump_candidates(
            self,
            candidates_pools,
            dump_dir=None,
            filename=None):

        result = {}
        for candidate_acc, candidate_pool in candidates_pools.items():
            candidates_sols = []
            for individual in candidate_pool:
                candidates_sols.append(individual.status())
            result[candidate_acc] = candidates_sols

        os.makedirs(os.path.dirname(dump_dir), exist_ok=True)
        filename = dump_dir + "/" + filename
        with open(filename, "w+") as fp:
            json.dump(result, fp, indent=4)
