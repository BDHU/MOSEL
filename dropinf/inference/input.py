import torch

class InputManager:
    
    def __init__(self, modality_names=None, excluded=None, gpu=True):
        r"""
        User specifies what modalities are used and their corresponding tensors in a batch
        
        modality_name: {modality_name: [modality's tensor]}
        excluded: [[modality combo that should be avoided]]
        """
        self.modality_names = modality_names
        self.num_mod = len(self.modality_names)
        self.excluded = excluded
        self.gpu = gpu
        self.mod_to_remove = self.__mod_to_remove()

    def __mod_to_remove(self):
        from itertools import chain, combinations
        def powerset(iterable):
            s = list(iterable)  # allows duplicate elements
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        
        mods_to_drop = powerset(self.modality_names.keys())
        return [i for i in mods_to_drop if len(i) < self.num_mod]  # remove full drop combo

    def get_mod_to_remove(self):
        return self.mod_to_remove

    def add(self, modality_name, mod_inputs):
        if modality_name in self.modality_names:
            raise ValueError("Modality name already exists")

        self.modality_names[modality_name] = mod_inputs

    def cuda(self, batch, device=None):
        for k, v in batch.items():
            if torch.is_tensor(v):
                cuda_dev = torch.device("cuda:" + str(device))
                batch[k] = v.to(cuda_dev)

        return batch


    def remove_mod(self, batch, removed_modality=None):
        r"""
        removed_modality: a list of modality names
        """
        if removed_modality is None or self.modality_names is None or removed_modality == () or removed_modality == "None":
            # return self.cuda(batch)
            return batch

        for rm in removed_modality:
            if rm == "None":
                continue
            del_tensors = self.modality_names[rm]
            for d_t in del_tensors:
                del batch[d_t]
                batch[d_t] = None

        return batch

    def slice(self, batch, start_idx=0, end_idx=1):
        batch_slice = {}
        for k, v in batch.items():
            if torch.is_tensor(v) or isinstance(v, list):
                batch_slice[k] = v[start_idx:end_idx]
            else:
                batch_slice[k] = v
        return batch_slice

    # user-defined batch splitting function
    def split_batch(self, batch, split_sections=None):
        raise NotImplementedError("Please define split_batch function")
