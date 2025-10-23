# 3D Model Constructor

A comprehensive 3D reconstruction application built with COLMAP and PyTorch, featuring a modern GUI for creating 3D models from image sequences.

## Features

- **Interactive GUI**: Modern Tkinter-based interface for easy 3D reconstruction
- **COLMAP Integration**: Full sparse and dense reconstruction pipeline
- **SuperPoint Features**: Advanced feature extraction using SuperPoint neural network
- **Multiple Visualization Options**: View point clouds, mesh models, and feature keypoints
- **Progress Tracking**: Real-time progress updates and detailed logging
- **Previous Model Loading**: Load and visualize previously created models

## Requirements

- Python 3.8+
- COLMAP (included in `bin/` directory)
- CUDA-compatible GPU (recommended for SuperPoint features)

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python app.py
```

### Basic Workflow

1. **Select Images**: Click "Select Images..." to choose your input images
2. **Start Reconstruction**: Click "Start 3D Reconstruction" and enter a folder name
3. **View Results**: Use the view buttons to visualize different aspects:
   - **Show SuperPoint Features**: View extracted keypoints and descriptors
   - **Show Point Cloud**: Visualize the dense 3D point cloud
   - **Show Mesh Model**: Display the reconstructed 3D mesh

### Advanced Features

- **Load Previous Model**: Load and visualize previously created models
- **View Latest Logs**: Check detailed processing logs
- **Progress Tracking**: Monitor reconstruction progress in real-time

## Project Structure

```
├── app.py                 # Main GUI application
├── models/                # Neural network models
│   ├── superpoint.py     # SuperPoint feature detector
│   └── weights/          # Pre-trained model weights
├── utils/                 # Utility modules
│   ├── colmap_sparse.py  # Sparse reconstruction
│   ├── colmap_dense.py   # Dense reconstruction
│   ├── feature_extraction.py  # Feature extraction
│   ├── image_processing.py    # Image utilities
│   └── visualization.py      # 3D visualization
├── bin/                   # COLMAP executables
├── images/               # Input images directory
└── outputs/              # Reconstruction outputs
```

## Reconstruction Pipeline

The application follows a complete 3D reconstruction pipeline:

1. **Image Preparation**: Copy and organize input images
2. **Sparse Reconstruction**: 
   - Feature extraction (SIFT)
   - Feature matching
   - Structure from Motion (SfM)
3. **Dense Reconstruction**:
   - Image undistortion
   - Patch match stereo
   - Point cloud fusion
4. **Mesh Generation**:
   - COLMAP mesher (primary)
   - Open3D fallback (if needed)

## Output Files

Each reconstruction creates a timestamped output directory containing:

- `sparse/`: Sparse reconstruction results (cameras, images, points)
- `dense/`: Dense point cloud (`fused.ply`)
- `mesh/`: 3D mesh model (`mesh.ply`)
- `database.db`: COLMAP database

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce image resolution or use CPU mode
2. **COLMAP Errors**: Ensure COLMAP executables are in the `bin/` directory
3. **Empty Point Clouds**: Check image quality and overlap between images

### Performance Tips

- Use high-quality, well-lit images
- Ensure good overlap between consecutive images
- Consider image resolution vs. processing time trade-offs

## Dependencies

See `requirements.txt` for the complete list of Python packages required.

## License

This project includes components from various sources. Please refer to individual file headers for specific licensing information.

## Contributing

This is a research/educational project. For issues or improvements, please create an issue in the repository.
