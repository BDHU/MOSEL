import json
import os


class Storage:

    def __init__(self):
        self.profiler_storage= {}

    def add(self, removed_mod=None, batch_size=None, record=None):
        self.profiler_storage[removed_mod] = {batch_size: record}

    def get(self, removed_mod=None, batch_size=None):
        if removed_mod is not None and batch_size is None:
            return self.profiler_storage[removed_mod]

        if removed_mod is not None and batch_size is not None:
            return self.profiler_storage[removed_mod][batch_size]

        raise ValueError("invalid query into Storage")

    def convert_metric(self, output_format="json"):
        result = {}
        for removed_mod, record in self.profiler_storage.items():
            if len(removed_mod) <= 0:
                removed_mod = "None"
            else:
                removed_mod = ",".join(removed_mod)
            dm_dict = {}
            for batch_size, recorder in record.items():
                dm_dict[batch_size] = recorder.to_dict()
            result[removed_mod] = dm_dict

        return result


    def dump(self, result_dir=None):
        if result_dir is None:
            result_dir = "./profiler_result"

        if not os.path.exists('./profiler_result'):
            os.makedirs(result_dir)

        with open(result_dir + "/profiler_result.json", "w+") as fp:
            json.dump(self.convert_metric(), fp, indent=4)

    def read(self, result_dir=None):
        if result_dir is None:
            result_dir = "./profiler_result"

        with open(result_dir + "/profiler_result.json", "r") as fp:
            data = json.load(fp)
            print(data)
        
