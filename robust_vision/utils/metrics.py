#!/usr/bin/env python3
""" Adapted from
https://github.com/facebookresearch/DeepSDF/blob/master/deep_sdf/metrics/chamfer.py"""

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def chamfer_distance(mesh1, mesh2, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance,
    i.e. the sum of both chamfers.
    mesh1: trimesh.base.Trimesh
    mesh2: trimesh.base.Trimesh
    """
    points_sampled1 = trimesh.sample.sample_surface(mesh1, num_mesh_samples)[0]
    points_sampled2 = trimesh.sample.sample_surface(mesh2, num_mesh_samples)[0]

    # one direction
    kd_tree1 = KDTree(points_sampled2)
    distances1, _ = kd_tree1.query(points_sampled1)
    chamfer1 = np.mean(np.square(distances1))

    # other direction
    kd_tree2 = KDTree(points_sampled1)
    distances2, _ = kd_tree2.query(points_sampled2)
    chamfer2 = np.mean(np.square(distances2))

    return chamfer1 + chamfer2
