from dropinf.optimizer.identifier import Identifier

class Individual:

    def __init__(
            self,
            policy=None,
            identity=None,
            latency=None,
            memory=None,
            avg_acc=None,
            throughput=None):
        r"""
        policy: {"dopped_mod" : batch, ...}
        identity: str(drop_mod:batch,..)
        """
        self.policy = policy
        self.latency = latency
        self.memory = memory
        self.avg_acc = avg_acc
        self.throughput = throughput
        self.identity = identity

        self._total_latency = 0
        self._total_max_memory = 0

    def clear_internal_states(self):
        self._total_latency = 0
        self._total_max_memory = 0

    def is_evaluated(self):
        if self.latency is None or self.memory is None:
            return False
        return True

    def update_policy(self, new_policy):
        new_policy = {k: v for k, v in new_policy.items() if
                v > 0} # clear 0-dim policy
        self.policy = new_policy

    def share_modality(self, other):
        for k, _ in other.policy.items():
            if k in self.policy:
                return True
        return False

    def batch_size(self):
        return sum(self.policy.values())

    def equal_policy(self, other):
        return self.identity == other.identity
        
    def print(self):
        policy_str = str(self.policy)
        identity_str = str(self.identity)
        lat_str = str(self.latency)
        mem_str = str(self.memory)
        acc_str = str(self.avg_acc)
        throughput_str = str(self.throughput)
        final_string = "dropped modality: " + policy_str + "\n" + \
            "id: " + identity_str + "\n" + \
            "average latency: " + lat_str + "\n" + \
            "average max memory: " + mem_str + "\n" + \
            "average accuracy: " + acc_str + "\n" + \
            "throughput: " + throughput_str + "\n"
        print(final_string)

    def status(self):
        """
        Return the dropping policy's status in a dict 
        """
        stats = {}
        stats["drop_policy"] = self.policy
        stats["average_accuracy"] = self.avg_acc
        stats["average_max_memory"] = self.memory
        stats["average_latency"] = self.latency
        stats["average_throughput"] = self.throughput
        return stats
