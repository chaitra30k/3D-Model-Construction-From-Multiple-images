import open3d as o3d
import numpy as np
import os

def create_simple_mesh_from_pointcloud(pointcloud_path, output_path):
    """Create a simple mesh from point cloud using Open3D"""
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        
        if len(pcd.points) == 0:
            raise ValueError("Point cloud is empty")
        
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        return True
        
    except Exception as e:
        print(f"Failed to create simple mesh: {e}")
        return False

def run_colmap_mesher(sparse_dir, dense_dir, mesh_dir):
    """Run COLMAP mesher as fallback"""
    import subprocess
    from utils.colmap_sparse import COLMAP_PATH
    
    try:
        # Use COLMAP's Poisson mesher
        subprocess.run([
            COLMAP_PATH, 'poisson_mesher',
            '--input_path', os.path.join(sparse_dir, '0'),
            '--output_path', os.path.join(mesh_dir, 'mesh.ply')
        ], check=True)
        return True
    except Exception as e:
        print(f"COLMAP mesher failed: {e}")
        return False 