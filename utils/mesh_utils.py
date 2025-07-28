"""
Util functions for trimesh
"""
from random import choices
import h5py
import os
import argparse
import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
import time
import pdb
import torch

import pytorch3d


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = trimesh.Trimesh(
            vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces
        )
    return mesh


def load_mesh(mesh_file: str, convert_scene_to_mesh=False):
    mesh = trimesh.load(mesh_file)
    if convert_scene_to_mesh:
        mesh = as_mesh(mesh)
    return mesh


def save_mesh(out_file, vertices: np.array, faces: np.array, vertex_colors=None):
    assert vertices.ndim == 2 and vertices.shape[-1] == 3
    out_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )
    out_mesh.export(out_file, include_color=vertex_colors is not None)


def get_dim_mesh(mesh, verbose=True):
    """Return dimensions of mesh"""
    verts = np.array(mesh.vertices)
    min_coords = np.min(verts, axis=0)
    max_coords = np.max(verts, axis=0)
    dim = np.max(max_coords - min_coords)
    return dim


def normalize_mesh(mesh: trimesh.Trimesh, norm_mesh_dim=2.0):
    verts = np.array(mesh.vertices)

    min_coords = np.min(verts, axis=0)
    max_coords = np.max(verts, axis=0)
    center = (min_coords + max_coords) / 2.0
    scale_factor = np.max(max_coords - min_coords) / norm_mesh_dim
    normalized_vertices = (verts - center) / scale_factor
    mesh.vertices = normalized_vertices


def apply_rot_to_mesh(mesh: trimesh.Trimesh, rot_mat: np.ndarray):
    verts = np.array(mesh.vertices)
    rotated_verts = np.dot(verts, rot_mat.T)
    mesh.vertices = rotated_verts


def convert_uv_texture_to_vertex_color(mesh: trimesh.Trimesh, subd_level=256):
    """Convert uv color to vertex color"""
    smallest_side = 2.0 / subd_level  # expand_rate [-1.0, 1.0], grid_resolution at subd_level
    subd_mesh = mesh.subdivide_to_size(smallest_side)
    # convert uv texture to vertex color
    vertices = subd_mesh.vertices
    vertex_uvs = np.array(subd_mesh.visual.uv)
    texture_image = subd_mesh.visual.material.image
    vertex_colors = trimesh.visual.uv_to_color(
        vertex_uvs, texture_image
    )
    vertex_colors = vertex_colors[:, :3]

    color_mesh = trimesh.Trimesh(
        vertices=subd_mesh.vertices,
        faces=subd_mesh.faces,
        vertex_colors=vertex_colors
    )
    return color_mesh


def convert_trimesh_to_pytorch3d_mesh(
    mesh_list: list,
    device='cpu',
    load_color=True,
    default_color=(0.8, 0.8, 0.8)
):
    """Convert list of trimesh to pytorch3d mesh"""
    verts_list = []
    faces_list = []
    verts_rgb_list = []
    for i, mesh in enumerate(mesh_list):
        verts = torch.from_numpy(np.array(mesh.vertices)).float()
        faces = torch.from_numpy(np.array(mesh.faces).astype(np.int64))
        if mesh.visual.vertex_colors is not None and load_color:
            vertex_colors = mesh.visual.vertex_colors
            vertex_colors = np.array(vertex_colors).astype(np.float32) / 255.0
            vertex_rgb = torch.from_numpy(vertex_colors[:, :3]).float()
        else:
            verts_rgb = torch.ones_like(verts) * torch.tensor(default_color)
            vertex_rgb = verts_rgb.float()

        verts_list.append(verts.to(device))
        faces_list.append(faces.to(device))
        verts_rgb_list.append(vertex_rgb.to(device))

    try:
        textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb_list)
        mesh = pytorch3d.structures.Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=textures
        )
    except:
        mesh = None
    return mesh


def convert_pytorch3d_mesh_to_trimesh(mesh):
    verts_list = mesh.verts_list()
    faces_list = mesh.faces_list()
    if mesh.textures is not None:
        verts_rgb_list = mesh.textures.verts_features_list()
    else:
        verts_rgb_list = None

    out_mesh_list = []
    for i in range(len(verts_list)):
        vertices = (verts_list[i]).detach().cpu().numpy().astype(np.float32)
        faces = (faces_list[i]).detach().cpu().numpy().astype(np.int64)
        if verts_rgb_list is not None:
            vertex_colors = verts_rgb_list[i] * 255.0
            vertex_colors = vertex_colors.detach().cpu().numpy().astype(np.uint8)
            out_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
        else:
            out_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
            )
        out_mesh_list.append(out_mesh)

    return out_mesh_list

