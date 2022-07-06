"""
Script to test the difference between jitted and non-jitted code.
"""
import jax
import jax.numpy as jnp
import os
import sys
import timeit

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from jat.graph import Node, Graph


def get_neighbors(node_list, cutoff_dist):
    """
    Build an adjacency matrix for the given nodes. Two nodes are considered
    neighbors if their `pos` vectors are within `cutoff_dist` of each other.

    Parameters
    ----------
    node_list: List[Node]
        List of Node objects
    cutoff_dist: float
        Neighbor cutoff distance

    Returns
    -------
    jnp.ndarray
        Node adjacency matrix
    """

    pos_array = jnp.asarray([node.pos for node in node_list])
    dists = jnp.asarray(
        [
            [
                jnp.linalg.norm(pos_array[i, :] - pos_array[j, :])
                for j in range(pos_array.shape[0])
            ]
            for i in range(pos_array.shape[0])
        ]
    )

    return dists <= cutoff_dist


def generate_nodes(key):
    node_list = []
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    for _ in range(1000):
        node_list.append(Node(jnp.zeros((1,)), jax.random.normal(subkey, (3,))))
        key, subkey = jax.random.split(key)

    print("generated nodes", flush=True)
    return node_list


# node_list = [
#     Node(jnp.zeros((1,)), jnp.asarray([0.0, 0.0, 0.0])),
#     Node(jnp.zeros((1,)), jnp.asarray([0.0, 1.0, 0.0])),
#     Node(jnp.zeros((1,)), jnp.asarray([0.0, 0.0, 1.0])),
# ]
# node_list = []
# key, subkey = jax.random.split(jax.random.PRNGKey(0))
# for _ in range(1000):
#     node_list.append(Node(jnp.zeros((1,)), jax.random.normal(subkey, (3,))))
#     key, subkey = jax.random.split(key)

# print(node_list[:5], flush=True)

# all_pos = jnp.asarray([node.pos for node in node_list])

setup_cmd = "node_list = generate_nodes(jax.random.PRNGKey(0))"
print("non-jit:")
print(
    timeit.timeit(
        "get_neighbors(node_list, 1.0)",
        setup=setup_cmd,
        number=1,
        globals=globals(),
    ),
    flush=True,
)
print("jit:")
print(
    timeit.timeit(
        "Graph.get_neighbors(node_list, 1.0)",
        setup=setup_cmd,
        number=1,
        globals=globals(),
    ),
    flush=True,
)
