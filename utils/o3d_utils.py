import numpy as np
import open3d as o3d


def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def visualize_point_cloud(
    points,
    colors=None,
    normals=None,
    show_frame=False,
    frame_size=1.0,
    frame_origin=(0, 0, 0)
):
    """Visualize a point cloud."""
    pc = np2pcd(points, colors, normals)
    geometries = [pc]
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries)
    return pc


def convert_np_to_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    verts_color: np.ndarray = None,
):
    """"""
    mesh = o3d.geometry.TriangleMesh()

    # Convert numpy arrays to Open3D format and set them to the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if verts_color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(verts_color)

    # Compute normals for the mesh
    mesh.compute_vertex_normals()
    return mesh
