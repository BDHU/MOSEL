class Database:

    def __init__(
            self,
            batch_list=None,
            accuracies=None,
            data=None):
        self.batch_list = batch_list
        self.accuracies = accuracies
        self.data = data
        self._prune()

    def __delitem__(self, key):
        self.data.pop(key)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _prune(self):
        
        for i, accs in enumerate(self.data):
            if accs[0] is None:
                continue
            highest_latency = 999999999999999
            index = len(accs) - 1
            for p in reversed(accs):
                if self.get_latency(p) <= highest_latency:
                    highest_latency = self.get_latency(p)
                else:
                    accs[index] = accs[index+1]
                index -= 1



        for i, accs in enumerate(self.data):
            # zero batch size has no policy
            if accs[0] is None:
                continue
            for p in accs:
                p[0] = {k: v for k, v in p[0].items() if v > 0}

            # todo ensure the highest accuracy is retrained, reverse order
            # acc set
            pruned = []
            for p in reversed(accs):
                pruned_acc_set = set([a[4] for a in pruned])
                if p[4] not in pruned_acc_set:
                    pruned.append(p)

            pruned.reverse()
            self.data[i] = pruned

        # remove redundancy policy
        # for i, accs in enumerate(self.data):
        #     if accs[i] is None:
        #         continue
        #     for p in accs:



    def get(self, acc_target=None, batch_size=None):
        accs = self.data[batch_size]
        for i, x in enumerate(accs):
            acc = self.get_acc(x)
            if acc >= acc_target:
                return i, self.data[batch_size][i]

        # Can't find accuracy
        return None, None

    def get_index(self, acc=None, batch_size=None):
        accs = self.data[batch_size]
        for i, x in enumerate(accs):
            if acc == self.get_acc(x):
                return batch_size, i

        raise ValueError("Invalid acc or batch_size")

    def get_by_b_policy(self, policy=None, batch_size=None):
        accs = self.data[batch_size]
        for i, x in enumerate(accs):
            p = self.get_mod_policy(x)
            if p == policy:
                return self.data[batch_size][i]

        raise ValueError("Invalid policy or batch_size")

    def get_max_acc_by_b(self, batch_size=None):
        return self.data[batch_size][-1]
    
    def get_b_metrics(self, batch_size=None):
        assert(batch_size is not None)
        return self.data[batch_size]


    def get_mod_policy(self, x):
        return x[0]

    def get_latency(self, x):
        return x[2]

    def get_acc_target(self, x):
        return x[3]

    def get_acc(self, x):
        return x[4]
