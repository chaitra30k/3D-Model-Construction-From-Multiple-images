import os
import shutil
import datetime
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from utils.colmap_sparse import run_colmap_sparse
from utils.colmap_dense import run_colmap_dense
from utils.mesh_generation import run_colmap_mesher
from utils.visualization import show_keypoints, show_point_cloud, show_mesh
from utils.feature_extraction import extract_superpoint_features
from utils.image_processing import load_image
import numpy as np

IMAGES_DIR = 'images'
OUTPUTS_DIR = 'outputs'

class ThreeDModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Model Constructor")
        # Start maximized
        self.root.state('zoomed')
        # self.root.geometry("1200x800")  # Remove fixed geometryyy
        self.root.resizable(True, True)

        self.image_paths = []
        self.latest_run_dir = None
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        self._create_widgets()
        self._update_log("Application started. Please select images to begin.")

    def _configure_styles(self):
        self.style.configure('Accent.TButton', font=('Arial', 12, 'bold'), foreground='white', background='#4CAF50', borderwidth=0, focusthickness=3, focuscolor='none')
        self.style.map('Accent.TButton', background=[('active', '#66BB6A'), ('disabled', '#A5D6A7')], foreground=[('disabled', '#EEEEEE')])
        self.style.configure('TButton', font=('Arial', 10), padding=6)
        self.style.map('TButton', background=[('active', '#e0e0e0')], foreground=[('disabled', '#a0a0a0')])
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabelframe', background='#f0f0f0')
        self.style.configure('TLabelframe.Label', font=('Arial', 12, 'bold'), background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'), background='#f0f0f0')
        self.style.configure('Status.TLabel', font=('Arial', 10, 'italic'), foreground='#333333', background='#f0f0f0')
        self.style.configure('TText', font=('Consolas', 9), background='#ecf0f1', foreground='#2c3e50')

    def _create_widgets(self):
        control_frame = ttk.LabelFrame(self.root, text="Image Selection & Reconstruction", padding="15 15 15 15")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=15, pady=10)
        upload_frame = ttk.Frame(control_frame)
        upload_frame.pack(fill=tk.X, pady=10)
        self.select_images_btn = ttk.Button(upload_frame, text="Select Images...", command=self._select_images)
        self.select_images_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        self.image_count_label = ttk.Label(upload_frame, text="No images selected.")
        self.image_count_label.pack(side=tk.LEFT, padx=10)
        self.image_list_preview = tk.Listbox(upload_frame, height=3, width=50, selectmode=tk.SINGLE, font=('Arial', 9), bg='#ffffff', fg='#333333')
        self.image_list_preview.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.image_list_preview_scrollbar = ttk.Scrollbar(upload_frame, orient=tk.VERTICAL, command=self.image_list_preview.yview)
        self.image_list_preview_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.image_list_preview.config(yscrollcommand=self.image_list_preview_scrollbar.set)
        # Feature type radio buttons (removed, only SIFT is used)
        # self.feature_type_var = tk.StringVar(value='sift')
        # feature_frame = ttk.Frame(control_frame)
        # feature_frame.pack(fill=tk.X, pady=5)
        # ttk.Label(feature_frame, text="Feature Type:").pack(side=tk.LEFT, padx=(0, 10))
        # ttk.Radiobutton(feature_frame, text="SIFT (Default)", variable=self.feature_type_var, value='sift').pack(side=tk.LEFT)
        self.start_reconstruction_btn = ttk.Button(control_frame, text="Start 3D Reconstruction", command=self._start_reconstruction, state=tk.DISABLED, style='Accent.TButton')
        self.start_reconstruction_btn.pack(pady=20, ipadx=30, ipady=15)
        self.progress_bar = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress_bar.pack(pady=10, fill=tk.X, padx=20)
        self.progress_bar.stop()
        self.viz_frame = ttk.LabelFrame(self.root, text="3D Visualization Output", padding="10 10 10 10")
        self.viz_frame.pack(expand=True, fill=tk.BOTH, padx=15, pady=10)
        self.visualization_canvas = tk.Canvas(self.viz_frame, bg="#2c3e50", highlightthickness=0, relief=tk.FLAT)
        self.visualization_canvas.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.visualization_canvas.create_text(self.visualization_canvas.winfo_width() / 2, self.visualization_canvas.winfo_height() / 2, text="Upload images and click 'Start 3D Reconstruction'\n\n3D Visualization Output Area", fill="#ecf0f1", font=('Arial', 18, 'bold'), justify=tk.CENTER, tags="viz_text_placeholder")
        self.visualization_canvas.bind("<Configure>", self._on_canvas_resize)
        view_options_frame = ttk.LabelFrame(self.root, text="View Options", padding="10 10 10 10")
        view_options_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)
        self.show_superpoint_btn = ttk.Button(view_options_frame, text="Show SuperPoint Features", command=self._show_superpoint, state=tk.DISABLED)
        self.show_superpoint_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        self.show_pointcloud_btn = ttk.Button(view_options_frame, text="Show Point Cloud", command=self._show_pointcloud, state=tk.DISABLED)
        self.show_pointcloud_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        self.show_mesh_btn = ttk.Button(view_options_frame, text="Show Mesh Model", command=self._show_mesh, state=tk.DISABLED)
        self.show_mesh_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        # Add button to load previous model
        self.load_model_btn = ttk.Button(view_options_frame, text="Load Previous Model", command=self._load_previous_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        # Add button to view latest logs
        self.view_logs_btn = ttk.Button(view_options_frame, text="View Latest Logs", command=self._view_latest_logs)
        self.view_logs_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        log_frame = ttk.LabelFrame(self.root, text="Application Log", padding="10 10 10 10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)
        self.status_label = ttk.Label(log_frame, text="Status: Ready", anchor=tk.W, style='Status.TLabel')
        self.status_label.pack(fill=tk.X, pady=(0, 5))
        
        # Add step details frame
        step_frame = ttk.Frame(log_frame)
        step_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Step progress display
        self.step_label = ttk.Label(step_frame, text="Step: Ready to start", anchor=tk.W, style='Status.TLabel')
        self.step_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Step progress indicator
        self.step_progress = ttk.Progressbar(step_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.step_progress.pack(side=tk.RIGHT, padx=(10, 0))
        self.step_progress['value'] = 0
        
        self.log_text = tk.Text(log_frame, height=6, state=tk.DISABLED, wrap=tk.WORD, bg="#ecf0f1", fg="#2c3e50", font=('Consolas', 9), relief=tk.FLAT)
        self.log_text.pack(fill=tk.X, expand=True)

    def _on_canvas_resize(self, event):
        self.visualization_canvas.coords("viz_text_placeholder", event.width / 2, event.height / 2)

    def _update_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _update_step(self, step_name, step_number=0, total_steps=6):
        """Update the current step display"""
        if step_number > 0:
            self.step_label.config(text=f"Step {step_number}/{total_steps}: {step_name}")
            progress = (step_number / total_steps) * 100
            self.step_progress['value'] = progress
        else:
            self.step_label.config(text=f"Step: {step_name}")
            self.step_progress['value'] = 0
        self.root.update_idletasks()

    def _update_substep(self, substep_name):
        """Update the current substep display"""
        current_text = self.step_label.cget("text")
        if ":" in current_text:
            base_step = current_text.split(":")[0] + ":"
            self.step_label.config(text=f"{base_step} {substep_name}")
        else:
            self.step_label.config(text=f"Substep: {substep_name}")
        self.root.update_idletasks()

    def _select_images(self):
        new_image_paths = filedialog.askopenfilenames(title="Select Images for 3D Reconstruction", filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        if new_image_paths:
            self.image_paths = list(new_image_paths)
            self.image_count_label.config(text=f"{len(self.image_paths)} images selected.")
            self.image_list_preview.delete(0, tk.END)
            for i, path in enumerate(self.image_paths):
                self.image_list_preview.insert(tk.END, os.path.basename(path))
                if i % 2 == 0:
                    self.image_list_preview.itemconfig(tk.END, {'bg': '#f5f5f5'})
            self.start_reconstruction_btn.config(state=tk.NORMAL)
            self._update_log(f"Selected {len(self.image_paths)} images for reconstruction.")
            self._clear_visualization_area("Images selected. Click 'Start 3D Reconstruction'")
            self._disable_view_buttons()
        else:
            self.image_list_preview.delete(0, tk.END)
            self.image_count_label.config(text="No images selected.")
            self.start_reconstruction_btn.config(state=tk.DISABLED)
            self._update_log("Image selection cancelled.")
            self._clear_visualization_area("Upload images and click 'Start 3D Reconstruction'\n\n3D Visualization Output Area")
            self._disable_view_buttons()

    def _start_reconstruction(self):
        if not self.image_paths:
            messagebox.showerror("Error", "Please select images first!")
            return
        # Ask user for output folder name
        from tkinter.simpledialog import askstring
        folder_name = askstring("Output Folder Name", "Enter a name for the output folder:")
        if not folder_name or not folder_name.strip():
            self._update_log("Reconstruction cancelled: No output folder name provided.")
            return
        self._output_folder_name = folder_name.strip()
        self._set_ui_processing_state()
        self._update_log("Starting 3D reconstruction process...")
        self.status_label.config(text="Status: Processing... Please wait.")
        self._clear_visualization_area("Processing... This may take a while.\n\n(Running local pipeline)")
        self.progress_bar.start(10)
        processing_thread = threading.Thread(target=self._run_reconstruction_in_background)
        processing_thread.start()

    def _run_reconstruction_in_background(self):
        try:
            # Prepare images directory
            self.root.after(0, lambda: self._update_step("Preparing image directory", 1))
            os.makedirs(IMAGES_DIR, exist_ok=True)
            for f in os.listdir(IMAGES_DIR):
                file_path = os.path.join(IMAGES_DIR, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            for path in self.image_paths:
                fname = os.path.basename(path)
                dest = os.path.join(IMAGES_DIR, fname)
                shutil.copy(path, dest)
            self.root.after(0, lambda: self._update_log(f"Copied {len(self.image_paths)} images to working directory"))
            
            # Prepare output directories
            self.root.after(0, lambda: self._update_step("Setting up output directories", 2))
            run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._output_folder_name}"
            run_dir = os.path.join(OUTPUTS_DIR, run_name)
            sparse_dir = os.path.join(run_dir, 'sparse')
            dense_dir = os.path.join(run_dir, 'dense')
            mesh_dir = os.path.join(run_dir, 'mesh')
            database_path = os.path.join(run_dir, 'database.db')
            os.makedirs(sparse_dir, exist_ok=True)
            os.makedirs(dense_dir, exist_ok=True)
            os.makedirs(mesh_dir, exist_ok=True)
            self.root.after(0, lambda: self._update_log(f"Created output directory: {run_name}"))
            
            # Sparse reconstruction
            self.root.after(0, lambda: self._update_step("Running sparse reconstruction", 3))
            self.root.after(0, lambda: self._update_substep("Feature extraction, matching, mapping"))
            self.root.after(0, lambda: self._update_log("Extracting SIFT features from images..."))
            run_colmap_sparse(IMAGES_DIR, sparse_dir, database_path)
            self.root.after(0, lambda: self._update_log("✓ Sparse reconstruction completed"))
            
            # Dense reconstruction
            self.root.after(0, lambda: self._update_step("Running dense reconstruction", 4))
            self.root.after(0, lambda: self._update_substep("Depth estimation, point cloud fusion"))
            self.root.after(0, lambda: self._update_log("  → Undistorting images for dense reconstruction..."))
            run_colmap_dense(sparse_dir, IMAGES_DIR, dense_dir)
            self.root.after(0, lambda: self._update_log(" Dense reconstruction completed"))
            
            # Mesh generation
            self.root.after(0, lambda: self._update_step("Generating mesh model", 5))
            self.root.after(0, lambda: self._update_substep("Meshing (COLMAP/Open3D)"))
            self.root.after(0, lambda: self._update_log("  → Running COLMAP mesher..."))

            try:
                from utils.mesh_generation import create_simple_mesh_from_pointcloud, run_colmap_mesher

                # Try COLMAP mesher first
                if run_colmap_mesher(sparse_dir, dense_dir, mesh_dir):
                    self.root.after(0, lambda: self._update_log(" COLMAP mesh generation completed"))
                else:
                    raise Exception("COLMAP mesher reported failure")
            except Exception as e:
                self.root.after(0, lambda: self._update_log(f" COLMAP mesher failed: {str(e)}"))
                self.root.after(0, lambda: self._update_log("   Trying Open3D mesh generation..."))

                # Try Open3D mesh generation
                try:
                    dense_ply = os.path.join(dense_dir, 'fused.ply')
                    mesh_ply = os.path.join(mesh_dir, 'mesh.ply')
                    if create_simple_mesh_from_pointcloud(dense_ply, mesh_ply):
                        self.root.after(0, lambda: self._update_log(" Open3D mesh generation completed"))
                    else:
                        raise Exception("Open3D mesh generation reported failure")
                except Exception as fallback_error:
                    self.root.after(0, lambda: self._update_log(f" Mesh generation failed: {str(fallback_error)}"))
                    raise
            
            self.latest_run_dir = run_dir
            self.root.after(0, lambda: self._update_step("Finalizing reconstruction", 6))
            self.root.after(0, self._reconstruction_finished_callback)
        except Exception as e:
            self.root.after(0, lambda: self._update_log(f" Error during reconstruction: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Pipeline failed:\n{e}"))
            self.latest_run_dir = None
            self.root.after(0, self._reset_ui_after_error)
        finally:
            self.root.after(0, self.progress_bar.stop)

    def _reconstruction_finished_callback(self):
        self._update_log(" 3D reconstruction complete!")
        self.status_label.config(text="Status: Done!")
        self._update_step("Reconstruction completed", 6, 6)
        self._enable_all_buttons()
        self.progress_bar.stop()
        messagebox.showinfo("Success", "3D Model Construction Completed! You can now view the models.")
        self._enable_view_buttons()
        self._show_mesh()  # Automatically show mesh

    def _reset_ui_after_error(self):
        self.status_label.config(text="Status: Error or Ready")
        self._update_step("Ready to start", 0, 6)
        self.progress_bar.stop()
        self._enable_all_buttons()
        self._disable_view_buttons()
        self._clear_visualization_area("Error during processing. Please try again.", fill_color="#e74c3c")
        self.start_reconstruction_btn.config(state=tk.NORMAL)

    def _set_ui_processing_state(self):
        self.start_reconstruction_btn.config(state=tk.DISABLED)
        self.select_images_btn.config(state=tk.DISABLED)
        self._disable_view_buttons()
        self._update_step("Starting reconstruction...", 0, 6)

    def _enable_all_buttons(self):
        self.start_reconstruction_btn.config(state=tk.NORMAL)
        self.select_images_btn.config(state=tk.NORMAL)

    def _enable_view_buttons(self):
        self.show_superpoint_btn.config(state=tk.NORMAL)  # Enable SuperPoint button
        self.show_pointcloud_btn.config(state=tk.NORMAL)
        self.show_mesh_btn.config(state=tk.NORMAL)

    def _disable_view_buttons(self):
        self.show_superpoint_btn.config(state=tk.DISABLED)
        self.show_pointcloud_btn.config(state=tk.DISABLED)
        self.show_mesh_btn.config(state=tk.DISABLED)

    def _clear_visualization_area(self, message, fill_color="#ecf0f1"):
        self.visualization_canvas.delete("all")
        self.root.update_idletasks()
        self.visualization_canvas.create_text(self.visualization_canvas.winfo_width() / 2, self.visualization_canvas.winfo_height() / 2, text=message, fill=fill_color, font=('Arial', 18, 'bold'), justify=tk.CENTER, tags="viz_text_placeholder")

    def _show_superpoint(self):
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected for SuperPoint feature extraction.")
            return
        
        try:
            self._update_step("Loading SuperPoint model", 0, 6)
            self._update_log("Loading SuperPoint model and extracting features...")
            self._clear_visualization_area("Loading SuperPoint model...", fill_color="#FFD700")
            
            # Use the first image for demonstration
            image_path = self.image_paths[0]
            self._update_substep(f"Extracting features from {os.path.basename(image_path)}")
            self._update_log(f"Extracting SuperPoint features from: {os.path.basename(image_path)}")
            
            # Extract features using SuperPoint
            keypoints, descriptors, scores = extract_superpoint_features(image_path)
            
            self._update_log(f"✓ Extracted {len(keypoints)} keypoints with SuperPoint")
            
            # Visualize the keypoints
            self._update_substep("Visualizing keypoints")
            show_keypoints(image_path, keypoints, scores)
            
            self._update_log("SuperPoint features visualized successfully.")
            self._clear_visualization_area("SuperPoint Features Displayed\n\n(Check the popup window for visualization)", fill_color="#FFD700")
            self._update_step("SuperPoint visualization complete", 0, 6)
            
        except Exception as e:
            error_msg = f"Failed to load SuperPoint model or extract features:\n{e}"
            self._update_log(f" {error_msg}")
            self._update_step("SuperPoint extraction failed", 0, 6)
            messagebox.showerror("SuperPoint Error", error_msg)
            self._clear_visualization_area("SuperPoint feature extraction failed", fill_color="#e74c3c")

    def _show_pointcloud(self):
        if not self.latest_run_dir:
            messagebox.showerror("Error", "No reconstruction output found.")
            return
        dense_ply = os.path.join(self.latest_run_dir, 'dense', 'fused.ply')
        if not os.path.exists(dense_ply):
            messagebox.showerror("Error", f"Point cloud file not found: {dense_ply}")
            return
        try:
            self._update_step("Loading point cloud", 0, 6)
            self._update_log("Loading point cloud for visualization...")
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(dense_ply)
            if len(pcd.points) == 0:
                messagebox.showinfo("Empty Point Cloud", "No points found in the dense point cloud.")
                return
            show_point_cloud(np.asarray(pcd.points), np.asarray(pcd.colors) if pcd.has_colors() else None)
            self._update_log(" Point cloud visualized successfully.")
            self._update_step("Point cloud visualization complete", 0, 6)
        except Exception as e:
            self._update_log(f" Failed to show point cloud: {str(e)}")
            self._update_step("Point cloud visualization failed", 0, 6)
            messagebox.showerror("Error", f"Failed to show point cloud:\n{e}")

    def _show_mesh(self):
        if not self.latest_run_dir:
            messagebox.showerror("Error", "No reconstruction output found.")
            return
        
        # Mesh file path (generated by COLMAP mesher or Open3D fallback)
        possible_mesh_files = [
            os.path.join(self.latest_run_dir, 'mesh', 'mesh.ply')
        ]
        
        mesh_ply = None
        for mesh_file in possible_mesh_files:
            if os.path.exists(mesh_file):
                mesh_ply = mesh_file
                break
        
        if not mesh_ply:
            messagebox.showerror("Error", f"Mesh file not found. Checked:\n" + "\n".join(possible_mesh_files))
            return
        
        try:
            self._update_step("Loading mesh model", 0, 6)
            self._update_log("Loading mesh model for visualization...")
            show_mesh(mesh_ply)
            self._update_log(" Mesh visualized successfully.")
            self._update_step("Mesh visualization complete", 0, 6)
        except Exception as e:
            self._update_log(f" Failed to show mesh: {str(e)}")
            self._update_step("Mesh visualization failed", 0, 6)
            messagebox.showerror("Error", f"Failed to show mesh:\n{e}")

    def _load_previous_model(self):
        from tkinter import filedialog
        self._update_step("Loading previous model", 0, 6)
        selected_dir = filedialog.askdirectory(title="Select a previous model run directory", initialdir=OUTPUTS_DIR)
        if not selected_dir:
            self._update_log("Previous model selection cancelled.")
            self._update_step("Ready to start", 0, 6)
            return
        # Check for expected structure (dense/fused.ply, mesh/mesh.ply, etc.)
        dense_ply = os.path.join(selected_dir, 'dense', 'fused.ply')
        mesh_ply = os.path.join(selected_dir, 'mesh', 'mesh.ply')
        if not (os.path.exists(dense_ply) and os.path.exists(mesh_ply)):
            self._update_log(f"Selected folder does not contain a valid model: {selected_dir}")
            self._clear_visualization_area("Selected folder is not a valid model run.", fill_color="#e74c3c")
            self._disable_view_buttons()
            self._update_step("Invalid model folder", 0, 6)
            return
        self.latest_run_dir = selected_dir
        self._update_log(f" Loaded previous model: {selected_dir}")
        self._enable_view_buttons()
        self._clear_visualization_area("Previous model loaded. Use the view buttons below.")
        self._update_step("Previous model loaded", 0, 6)

    def _view_latest_logs(self):
        """Display only the application's logs"""
        # Get application logs
        app_logs = self.log_text.get(1.0, tk.END).strip()
        if not app_logs:
            messagebox.showinfo("No Logs", "No logs available to view.")
            return

        # Create log dialog
        log_dialog = tk.Toplevel(self.root)
        log_dialog.title("Latest Logs")
        log_dialog.geometry("1000x700")
        log_dialog.resizable(True, True)

        # Single text area with application logs
        text_widget = tk.Text(log_dialog, font=('Consolas', 9), bg="#ecf0f1", fg="#2c3e50", wrap=tk.WORD)
        text_widget.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        text_widget.insert(tk.END, app_logs)
        text_widget.config(state=tk.DISABLED)

        log_dialog.transient(self.root)
        log_dialog.grab_set()


if __name__ == "__main__":
    root = tk.Tk()
    app = ThreeDModelApp(root)
    root.mainloop() 