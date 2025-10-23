import subprocess
import os

COLMAP_PATH = r'D:\colmap-main\bin\colmap.exe'

def run_colmap_dense(sparse_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Image undistorter
    print("Running COLMAP image undistortion...")
    subprocess.run([
        COLMAP_PATH, 'image_undistorter',
        '--image_path', image_dir,
        '--input_path', os.path.join(sparse_dir, '0'),
        '--output_path', output_dir,
        '--output_type', 'COLMAP'
    ], check=True)
    print("✓ Image undistortion completed")
    
    # Patch match stereo with higher density
    print("Running COLMAP patch match stereo...")
    subprocess.run([
        COLMAP_PATH, 'patch_match_stereo',
        '--workspace_path', output_dir,
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true',
        '--PatchMatchStereo.max_image_size', '4000',
        '--PatchMatchStereo.window_radius', '7',
        '--PatchMatchStereo.num_samples', '20',
        '--PatchMatchStereo.num_iterations', '1'
    ], check=True)
    print("✓ Patch match stereo completed")
    
    # Stereo fusion with lower min_num_pixels for more points
    print("Running COLMAP stereo fusion...")
    subprocess.run([
        COLMAP_PATH, 'stereo_fusion',
        '--workspace_path', output_dir,
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', os.path.join(output_dir, 'fused.ply'),
        '--StereoFusion.min_num_pixels', '3'
    ], check=True)
    print("✓ Stereo fusion completed") 