import jax

class PRNGSequence:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)
        
    def __next__(self):
        self.key, key = jax.random.split(self.key)
        return key