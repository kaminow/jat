import jax
import jax.numpy as jnp
from typing import NamedTuple


class Node(NamedTuple):
    features: jnp.ndarray
    pos: jnp.ndarray


class Graph:
    @staticmethod
    @jax.jit
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
