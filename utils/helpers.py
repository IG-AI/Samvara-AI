import os
import shutil
import h5py

def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"{file_path} removed successfully."
        return f"{file_path} does not exist."
    except Exception as e:
        return f"Error removing {file_path}: {str(e)}"

def safe_remove_hdf5_dataset(file_path, dataset_name):
    try:
        with h5py.File(file_path, "a") as f:
            if dataset_name in f:
                del f[dataset_name]
                return f"Dataset {dataset_name} removed from {file_path}."
            return f"Dataset {dataset_name} not found in {file_path}."
    except Exception as e:
        return f"Error removing dataset {dataset_name}: {str(e)}"

def ensure_directory_exists_and_writable(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if not os.access(directory_path, os.W_OK):
        return f"Permission denied: {directory_path} is not writable."
    return f"Directory {directory_path} exists and is writable."

def clear_existing_checkpoints(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for file in os.listdir(checkpoint_dir):
        os.remove(os.path.join(checkpoint_dir, file))
    return f"Cleared existing checkpoints in {checkpoint_dir}."
