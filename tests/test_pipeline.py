import unittest
import tempfile
import os
import shutil
import subprocess
import sys
from unittest.mock import patch, MagicMock

# Import the pipeline components to test
import sys
sys.path.append('..')
from models.run_pipeline import parse_args, run_data_preparation, run_training, run_evaluation


class TestPipelineComponents(unittest.TestCase):
    """Test cases for pipeline components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy data files
        self.create_dummy_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_dummy_data(self):
        """Create dummy data files for testing."""
        # Create dummy CSV files
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        data = pd.DataFrame({
            'unit': [1, 1, 1, 2, 2, 2],
            'cycle': [1, 2, 3, 1, 2, 3],
            'op_setting_1': [0.5, 0.6, 0.7, 0.4, 0.5, 0.6],
            'op_setting_2': [0.3, 0.4, 0.5, 0.2, 0.3, 0.4],
            'op_setting_3': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            'sensor_1': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            'sensor_2': [0.2, 0.3, 0.4, 0.2, 0.3, 0.4],
            'RUL': [100, 99, 98, 95, 94, 93],
            'dataset_id': ['FD001'] * 6
        })
        
        # Add more sensor columns
        for i in range(3, 22):
            data[f'sensor_{i}'] = np.random.randn(6)
        
        # Save to CSV files
        data.to_csv(os.path.join(self.temp_dir, 'processed_train.csv'), index=False)
        data.to_csv(os.path.join(self.temp_dir, 'processed_val.csv'), index=False)
        data.to_csv(os.path.join(self.temp_dir, 'processed_test.csv'), index=False)
    
    def test_parse_args(self):
        """Test argument parsing."""
        # Test with default arguments
        with patch('sys.argv', ['run_pipeline.py']):
            args = parse_args()
            self.assertEqual(args.max_epochs, 50)
            self.assertEqual(args.batch_size, 64)
            self.assertEqual(args.window_size, 30)
            self.assertEqual(args.horizon, 1)
            self.assertFalse(args.skip_data_prep)
            self.assertFalse(args.skip_training)
            self.assertFalse(args.skip_evaluation)
            self.assertFalse(args.use_simple_model)
    
    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        with patch('sys.argv', [
            'run_pipeline.py',
            '--max_epochs', '10',
            '--batch_size', '32',
            '--window_size', '20',
            '--horizon', '2',
            '--use_simple_model'
        ]):
            args = parse_args()
            self.assertEqual(args.max_epochs, 10)
            self.assertEqual(args.batch_size, 32)
            self.assertEqual(args.window_size, 20)
            self.assertEqual(args.horizon, 2)
            self.assertTrue(args.use_simple_model)
    
    @patch('subprocess.run')
    def test_run_data_preparation(self, mock_run):
        """Test data preparation step."""
        mock_run.return_value = MagicMock(returncode=0)
        
        args = MagicMock()
        args.window_size = 30
        args.horizon = 1
        args.data_dir = 'transformer_data'
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            run_data_preparation(args)
            
            # Check if subprocess.run was called
            mock_run.assert_called_once()
            
            # Check the command that was run
            cmd = mock_run.call_args[0][0]
            self.assertIn('prepare_windows.py', cmd[1])
            self.assertIn('--window_size', cmd)
            self.assertIn('30', cmd)
            self.assertIn('--horizon', cmd)
            self.assertIn('1', cmd)
            
        finally:
            os.chdir(original_cwd)
    
    @patch('subprocess.run')
    def test_run_training_simple_model(self, mock_run):
        """Test training step with simple model."""
        mock_run.return_value = MagicMock(returncode=0)
        
        args = MagicMock()
        args.use_simple_model = True
        args.data_dir = 'transformer_data'
        args.batch_size = 32
        args.hidden_size = 64
        args.max_epochs = 5
        args.learning_rate = 0.001
        args.ckpt_dir = 'checkpoints'
        
        run_training(args)
        
        # Check if subprocess.run was called
        mock_run.assert_called_once()
        
        # Check the command that was run
        cmd = mock_run.call_args[0][0]
        self.assertIn('train_simple_lstm.py', cmd[1])
        self.assertIn('--batch_size', cmd)
        self.assertIn('32', cmd)
    
    @patch('subprocess.run')
    def test_run_training_tft_model(self, mock_run):
        """Test training step with TFT model."""
        mock_run.return_value = MagicMock(returncode=0)
        
        args = MagicMock()
        args.use_simple_model = False  # Use TFT
        args.data_dir = 'transformer_data'
        args.batch_size = 32
        args.hidden_size = 64
        args.max_epochs = 5
        args.learning_rate = 0.001
        args.ckpt_dir = 'checkpoints'
        
        run_training(args)
        
        # Check if subprocess.run was called
        mock_run.assert_called_once()
        
        # Check the command that was run
        cmd = mock_run.call_args[0][0]
        self.assertIn('train_tft.py', cmd[1])
        self.assertIn('--batch_size', cmd)
        self.assertIn('32', cmd)
    
    @patch('subprocess.run')
    def test_run_evaluation_simple_model(self, mock_run):
        """Test evaluation step with simple model."""
        mock_run.return_value = MagicMock(returncode=0)
        
        args = MagicMock()
        args.use_simple_model = True
        args.data_dir = 'transformer_data'
        args.batch_size = 32
        args.ckpt_dir = 'checkpoints'
        args.eval_dir = 'evaluation_plots'
        
        # Create dummy checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(os.path.join(args.ckpt_dir, 'lstm-epoch=01-val_loss=1000.0000.ckpt'), 'w') as f:
            f.write('dummy checkpoint')
        
        run_evaluation(args)
        
        # Check if subprocess.run was called
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_evaluation_tft_model(self, mock_run):
        """Test evaluation step with TFT model."""
        mock_run.return_value = MagicMock(returncode=0)
        
        args = MagicMock()
        args.use_simple_model = False  # Use TFT
        args.data_dir = 'transformer_data'
        args.batch_size = 32
        args.ckpt_dir = 'checkpoints'
        args.eval_dir = 'evaluation_plots'
        
        # Create dummy checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(os.path.join(args.ckpt_dir, 'tft-epoch=01-val_loss=1000.0000.ckpt'), 'w') as f:
            f.write('dummy checkpoint')
        
        run_evaluation(args)
        
        # Check if subprocess.run was called
        mock_run.assert_called_once()
        
        # Check the command that was run
        cmd = mock_run.call_args[0][0]
        self.assertIn('evaluate_tft.py', cmd[1])


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end pipeline tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal data structure
        self.create_minimal_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_minimal_data(self):
        """Create minimal data for end-to-end testing."""
        # Create data directory
        data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create minimal CMAPSS data files
        train_data = "1 1 0.5 0.3 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1\n"
        train_data += "1 2 0.6 0.4 0.2 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2\n"
        
        test_data = "1 1 0.5 0.3 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1\n"
        
        rul_data = "10\n"
        
        # Save files
        with open(os.path.join(data_dir, 'train_FD001.txt'), 'w') as f:
            f.write(train_data)
        
        with open(os.path.join(data_dir, 'test_FD001.txt'), 'w') as f:
            f.write(test_data)
        
        with open(os.path.join(data_dir, 'RUL_FD001.txt'), 'w') as f:
            f.write(rul_data)
    
    def test_preprocessing_script(self):
        """Test that preprocessing script can run."""
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Test preprocessing with minimal data
            from preprocess_data import prepare_cmapss_data
            
            result = prepare_cmapss_data(
                dataset_ids=['FD001'],
                val_size=0.3
            )
            
            # Check that we got the expected structure
            self.assertIn('train', result)
            self.assertIn('val', result)
            self.assertIn('test', result)
            self.assertIn('scaler', result)
            
            # Check that data is not empty
            self.assertGreater(len(result['train']), 0)
            # With small dataset, validation might be empty, so check total size
            total_samples = len(result['train']) + len(result['val'])
            self.assertGreater(total_samples, 0)
            self.assertGreater(len(result['test']), 0)
            
        finally:
            os.chdir(original_cwd)
    
    def test_model_imports(self):
        """Test that all model modules can be imported."""
        try:
            from models.simple_lstm_model import SimpleLSTM
            from models.tft_model import TemporalFusionTransformer
            from models.tft_data_module import CMAPSSDataModule
            from models.run_pipeline import parse_args
            
            # Test model creation
            lstm_model = SimpleLSTM(input_size=24, hidden_size=32)
            self.assertIsNotNone(lstm_model)
            
            tft_model = TemporalFusionTransformer(
                num_time_varying_real_vars=24,
                hidden_size=32
            )
            self.assertIsNotNone(tft_model)
            
        except ImportError as e:
            self.fail(f"Failed to import model modules: {e}")


class TestPipelineErrorHandling(unittest.TestCase):
    """Test error handling in pipeline."""
    
    @patch('subprocess.run')
    def test_subprocess_error_handling(self, mock_run):
        """Test handling of subprocess errors."""
        # Mock subprocess to raise an error
        mock_run.side_effect = subprocess.CalledProcessError(1, ['python', 'script.py'])
        
        args = MagicMock()
        args.window_size = 30
        args.horizon = 1
        args.data_dir = 'transformer_data'
        
        # Should raise CalledProcessError
        with self.assertRaises(subprocess.CalledProcessError):
            run_data_preparation(args)
    
    def test_missing_data_files(self):
        """Test handling of missing data files."""
        # Create empty temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                from preprocess_data import prepare_cmapss_data
                
                # Should raise FileNotFoundError
                with self.assertRaises(FileNotFoundError):
                    prepare_cmapss_data(dataset_ids=['FD001'])
                    
            finally:
                os.chdir(original_cwd)
                
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main() 