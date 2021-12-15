import jax
import jax.numpy as jnp
from functools import partial

def tree_index(tree, idx):
    """Index a stack object."""
    return jax.tree_map(lambda a: a[idx], tree)


def tree_multimap_rand(fn, key, *trees, **kwargs):
    """Like jax.tree_multimap, but first argument of fn is key."""
    leaves, treedef = jax.tree_flatten(trees[0])
    keys = jax.random.split(key, len(leaves))
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in trees[1:]]
    
    if len(kwargs) != 0:
        fn = partial(fn, **kwargs)
    
    return treedef.unflatten(fn(*args) for args in zip(keys, *all_leaves))


def stack_trees(trees):
    """Create single stacked object representing a list of objects.
    
    Inverse of unstack_tree.
    """
    multi = jax.tree_multimap(lambda *par: jnp.stack(par), *trees)
    return multi

def unstack_tree(tree):
    """Convert stacked object to list of objects.
    
    Inverse of stack_trees.
    """
    arr, treedef = jax.tree_flatten(tree)
    return [treedef.unflatten(a) for a in zip(*map(list, arr))]