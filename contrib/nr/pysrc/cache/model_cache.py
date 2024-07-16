class ModelCache:
    def __init__(self):
        self.cache = {}
        self.next_model_id = 1

    def add_model(self, model_name, model):
        if model_name not in self.cache:
            self.cache[model_name] = {}
        model_id = self.next_model_id
        self.cache[model_name][model_id] = model
        self.next_model_id += 1
        return model_id

    def get_model(self, model_name, model_id):
        if model_name in self.cache:
            return self.cache[model_name].get(model_id, None)
        return None

    def clear(self):
        self.cache.clear()
        self.next_model_id = 1
