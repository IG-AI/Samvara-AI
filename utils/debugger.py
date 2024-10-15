# utils/debugger.py

import os
import subprocess
import random
import tensorflow as tf
import unittest
from unittest.mock import patch, MagicMock
from utils.helpers import safe_remove, safe_remove_hdf5_dataset, ensure_directory_exists_and_writable, clear_existing_checkpoints
from utils.gpu_monitor import start_gpu_monitoring

def some_function():
    from utils.debugger import check_directory_permissions
    # function body

class TestDebuggerFunctions(unittest.TestCase):

    @patch("os.path.exists")
    @patch("os.access")
    def test_check_directory_permissions(self, mock_access, mock_exists):
        # Test case where directory does not exist
        mock_exists.return_value = False
        result = check_directory_permissions("/non/existent/path")
        self.assertEqual(result, "Directory /non/existent/path does not exist.")
        
        # Test case where directory is not writable
        mock_exists.return_value = True
        mock_access.return_value = False
        result = check_directory_permissions("/non/writable/path")
        self.assertEqual(result, "Permission denied: /non/writable/path is not writable.")
        
        # Test case where directory is accessible and writable
        mock_access.return_value = True
        result = check_directory_permissions("/writable/path")
        self.assertEqual(result, "Directory /writable/path is accessible and writable.")
    
    @patch("subprocess.run")
    def test_check_docker_container_exists(self, mock_run):
        # Test case where container exists
        mock_run.return_value.stdout.decode.return_value = "existing_container"
        result = check_docker_container_exists("existing_container")
        self.assertEqual(result, "Container with the name existing_container already exists.")
        
        # Test case where container does not exist
        mock_run.return_value.stdout.decode.return_value = ""
        result = check_docker_container_exists("non_existent_container")
        self.assertEqual(result, "Container non_existent_container does not exist.")
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_docker_container_exists("error_container")
        self.assertEqual(result, "Error checking Docker container: error")

    @patch("subprocess.run")
    def test_check_docker_running(self, mock_run):
        # Test case where Docker is running
        mock_run.return_value.stdout.decode.return_value = "active"
        result = check_docker_running()
        self.assertEqual(result, "Docker service is running.")
        
        # Test case where Docker is not running
        mock_run.return_value.stdout.decode.return_value = "inactive"
        result = check_docker_running()
        self.assertEqual(result, "Docker service is not running.")
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_docker_running()
        self.assertEqual(result, "Error checking Docker service: error")

    @patch("subprocess.run")
    def test_check_python_packages(self, mock_run):
        # Test case where package is installed
        mock_run.return_value.returncode = 0
        result = check_python_packages("container", "package")
        self.assertEqual(result, "Package package is installed in container container.")
        
        # Test case where package is missing
        mock_run.return_value.returncode = 1
        result = check_python_packages("container", "missing_package")
        self.assertEqual(result, "Package missing_package is missing inside the container container.")
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_python_packages("container", "error_package")
        self.assertEqual(result, "Error checking package error_package: error")

    @patch("tensorflow.config.experimental.list_physical_devices")
    def test_check_gpu_availability(self, mock_list_physical_devices):
        # Test case where GPUs are available
        mock_list_physical_devices.return_value = ["GPU1", "GPU2"]
        result = check_gpu_availability()
        self.assertEqual(result, "Available GPUs: 2")
        
        # Test case where no GPUs are available
        mock_list_physical_devices.return_value = []
        result = check_gpu_availability()
        self.assertEqual(result, "No GPU detected.")

    @patch("subprocess.run")
    def test_check_docker_volume(self, mock_run):
        # Test case where volume is mounted correctly
        mock_run.return_value.returncode = 0
        result = check_docker_volume("container", "/volume/path")
        self.assertEqual(result, "Volume /volume/path is mounted correctly in container.")
        
        # Test case where volume is not mounted correctly
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr.decode.return_value = "error"
        result = check_docker_volume("container", "/volume/error")
        self.assertEqual(result, "Volume /volume/error not mounted correctly in container: error")
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_docker_volume("container", "/volume/exception")
        self.assertEqual(result, "Error checking Docker volume /volume/exception: error")

    @patch("subprocess.run")
    def test_check_docker_image_integrity(self, mock_run):
        # Test case where Docker image integrity is intact
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout.decode.return_value = "layers"
        result = check_docker_image_integrity("image_name")
        self.assertEqual(result, "Docker image image_name integrity is intact.")
        
        # Test case where Docker image has issues with layers
        mock_run.return_value.stdout.decode.return_value = ""
        result = check_docker_image_integrity("image_name")
        self.assertEqual(result, "Docker image image_name has issues with layers.")
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_docker_image_integrity("image_name")
        self.assertEqual(result, "Error checking Docker image integrity: error")

    @patch("os.chmod")
    def test_randomize_directory_permissions(self, mock_chmod):
        # Test case where permissions are set successfully
        result = randomize_directory_permissions("/path")
        self.assertIn("Set random permissions", result)
        
        # Test case where an error occurs
        mock_chmod.side_effect = Exception("error")
        result = randomize_directory_permissions("/path")
        self.assertEqual(result, "Error setting random permissions for /path: error")

    @patch("subprocess.run")
    def test_random_command_execution(self, mock_run):
        # Test case where command succeeds
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout.decode.return_value = "success"
        result = random_command_execution("container")
        self.assertIn("succeeded in container", result)
        
        # Test case where command fails
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr.decode.return_value = "error"
        result = random_command_execution("container")
        self.assertIn("failed in container", result)
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = random_command_execution("container")
        self.assertEqual(result, "Error executing random command: error")

    @patch("subprocess.run")
    def test_random_file_operations(self, mock_run):
        # Test case where file is created successfully
        mock_run.return_value.returncode = 0
        result = random_file_operations("container", "/base/path")
        self.assertIn("File /base/path/test_file_", result)
        
        # Test case where file creation fails
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr.decode.return_value = "error"
        result = random_file_operations("container", "/base/error")
        self.assertIn("Failed to create file", result)
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = random_file_operations("container", "/base/exception")
        self.assertEqual(result, "Error during file operation in container: error")

    @patch("subprocess.run")
    def test_check_tensorflow_env(self, mock_run):
        # Test case where TensorFlow environment is correctly configured
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout.decode.return_value = "configured"
        result = check_tensorflow_env()
        self.assertIn("TensorFlow environment is correctly configured", result)
        
        # Test case where TensorFlow environment check fails
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr.decode.return_value = "error"
        result = check_tensorflow_env()
        self.assertIn("TensorFlow environment check failed", result)
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_tensorflow_env()
        self.assertEqual(result, "Error checking TensorFlow environment: error")

    @patch("subprocess.run")
    def test_check_file_access(self, mock_run):
        # Test case where file access test succeeds
        mock_run.return_value.returncode = 0
        result = check_file_access("container", "/file/path")
        self.assertIn("File access test succeeded", result)
        
        # Test case where write access fails
        mock_run.side_effect = [subprocess.CompletedProcess(args=[], returncode=1, stderr=b"error"), MagicMock()]
        result = check_file_access("container", "/file/write_error")
        self.assertIn("Write access failed", result)
        
        # Test case where read access fails
        mock_run.side_effect = [MagicMock(), subprocess.CompletedProcess(args=[], returncode=1, stderr=b"error")]
        result = check_file_access("container", "/file/read_error")
        self.assertIn("Read access failed", result)
        
        # Test case where an error occurs
        mock_run.side_effect = Exception("error")
        result = check_file_access("container", "/file/exception")
        self.assertEqual(result, "Error during file access test for /file/exception: error")

    @patch("subprocess.run")
    def test_check_docker_env_vars(self, mock_run):
        # Test case where environment variables are checked successfully
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout.decode.return_value = "env_vars"
        result = check_docker_env_vars("container")
        self.assertIn("Environment variables in container", result)
        
        # Test case where an error occurs
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr.decode.return_value = "error"
        result = check_docker_env_vars("container")
        self.assertIn("Error checking environment variables in container", result)
        
        # Test case where an exception occurs
        mock_run.side_effect = Exception("error")
        result = check_docker_env_vars("container")
        self.assertEqual(result, "Error checking Docker environment variables: error")

    @patch("subprocess.run")
    def test_check_gpu_memory_usage(self, mock_run):
        # Test case where GPU memory usage is checked successfully
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout.decode.return_value = "memory_usage"
        result = check_gpu_memory_usage()
        self.assertIn("GPU memory usage", result)
        
        # Test case where an error occurs
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr.decode.return_value = "error"
        result = check_gpu_memory_usage()
        self.assertIn("Error checking GPU memory usage", result)
        
        # Test case where an exception occurs
        mock_run.side_effect = Exception("error")
        result = check_gpu_memory_usage()
        self.assertEqual(result, "Error checking GPU memory usage: error")

    @patch("tensorflow.keras.Model.load_weights")
    def test_test_model_loading(self, mock_load_weights):
        from models.material_layers import build_material_model
        from models.immaterial_layers import build_immaterial_model
        
        # Test case where model loading succeeds
        result = test_model_loading("Material Layer Model", build_material_model, "/path/to/model")
        self.assertIn("Model Material Layer Model loaded successfully", result)
        
        # Test case where model loading fails
        mock_load_weights.side_effect = Exception("error")
        result = test_model_loading("Immaterial Layer Model", build_immaterial_model, "/path/to/model")
        self.assertIn("Error loading model Immaterial Layer Model", result)

    @patch("tensorflow.keras.Model.__call__")
    def test_test_model_inference(self, mock_call):
        from models.material_layers import build_material_model
        from models.immaterial_layers import build_immaterial_model
        
        # Test case where model inference succeeds
        mock_call.return_value.shape = (10, 3)
        result = test_model_inference("Material Layer Model", build_material_model, [(10, 3), (10, 3)])
        self.assertIn("Inference on Material Layer Model succeeded", result)
        
        # Test case where model inference fails
        mock_call.side_effect = Exception("error")
        result = test_model_inference("Immaterial Layer Model", build_immaterial_model, [(10, 2), (10, 2)])
        self.assertIn("Error during inference on Immaterial Layer Model", result)

    @patch("tensorflow.keras.Model.save_weights")
    def test_test_model_saving(self, mock_save_weights):
        from models.material_layers import build_material_model
        from models.immaterial_layers import build_immaterial_model
        
        # Test case where model saving succeeds
        result = test_model_saving("Material Layer Model", build_material_model, "/path/to/save")
        self.assertIn("Model Material Layer Model saved successfully", result)
        
        # Test case where model saving fails
        mock_save_weights.side_effect = Exception("error")
        result = test_model_saving("Immaterial Layer Model", build_immaterial_model, "/path/to/save")
        self.assertIn("Error saving model Immaterial Layer Model", result)

    @patch("tensorflow.keras.Model.fit")
    def test_test_model_training_accuracy(self, mock_fit):
        from models.material_layers import build_material_model
        from models.immaterial_layers import build_immaterial_model
        
        # Test case where model training succeeds
        mock_fit.return_value.history = {'accuracy': [0.8, 0.9]}
        result = test_model_training_accuracy("Material Layer Model", build_material_model, [(10, 3), (10, 3)])
        self.assertIn("Material Layer Model training completed successfully", result)
        
        # Test case where model training fails
        mock_fit.side_effect = Exception("error")
        result = test_model_training_accuracy("Immaterial Layer Model", build_immaterial_model, [(10, 2), (10, 2)])
        self.assertIn("Error during training for Immaterial Layer Model", result)

    @patch("tensorflow.keras.Model.get_config")
    @patch("tensorflow.keras.Model.summary")
    def test_test_model_config(self, mock_summary, mock_get_config):
        from models.material_layers import build_material_model
        from models.immaterial_layers import build_immaterial_model
        
        # Test case where model config and summary are retrieved successfully
        mock_get_config.return_value = {"config": "test"}
        mock_summary.side_effect = lambda print_fn: print_fn("model_summary")
        result = test_model_config("Material Layer Model", build_material_model)
        self.assertIn("Material Layer Model config and summary:", result)
        
        # Test case where model config retrieval fails
        mock_get_config.side_effect = Exception("error")
        result = test_model_config("Immaterial Layer Model", build_immaterial_model)
        self.assertIn("Error getting config or summary for Immaterial Layer Model", result)

    @patch("tensorflow.keras.Model.__call__")
    def test_test_randomized_model_input(self, mock_call):
        from models.material_layers import build_material_model
        from models.immaterial_layers import build_immaterial_model
        
        # Test case where randomized model input succeeds
        mock_call.return_value.shape = (10, 3)
        result = test_randomized_model_input("Material Layer Model", build_material_model)
        self.assertIn("Material Layer Model handled randomized input", result)
        
        # Test case where randomized model input fails
        mock_call.side_effect = Exception("error")
        result = test_randomized_model_input("Immaterial Layer Model", build_immaterial_model)
        self.assertIn("Error with randomized input on Immaterial Layer Model", result)

if __name__ == '__main__':
    unittest.main()
