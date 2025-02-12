class Identifier:
    r"""
    If we also enforce submission order, identifier might not be neccesary.
    Identity class is used to translate dropped modality name into integer number
    Example:
        drop audio : 0
        drop video : 1
        no drop    : 2
        0 : drop audio
        1 : drop video
    """

    def __init__(self, all_dropped_mod=None):
        self.dropped_mod_to_id = {}
        self.id_to_dropped_mod = {}

        # modality name : id
        for i, k in enumerate(all_dropped_mod):
            self.dropped_mod_to_id[k] = i
        # id: modality_name
        for mod_name, mod_id in self.dropped_mod_to_id.items():
            self.id_to_dropped_mod[mod_id] = mod_name

    def get_mod_name(self, mod_id=None):
        assert(mod_id is not None)
        return self.id_to_dropped_mod[mod_id]

    def get_mod_id(self, mod_name=None):
        assert(mod_name is not None)
        return self.dropped_mod_to_id[mod_name]

    def print_name_to_id(self):
        print(self.dropped_mod_to_id)

    def print_id_to_name(self):
        print(self.id_to_dropped_mod)
