import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_keypoints(image_path, keypoints, scores=None):
    """Display keypoints on an image"""
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    
    # Plot keypoints
    if scores is not None:
        # Color keypoints by their scores
        scatter = plt.scatter(keypoints[:, 0], keypoints[:, 1], 
                            c=scores, cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, label='Keypoint Score')
    else:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=20, alpha=0.7)
    
    plt.title(f'SuperPoint Keypoints: {len(keypoints)} detected')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def show_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh]) 