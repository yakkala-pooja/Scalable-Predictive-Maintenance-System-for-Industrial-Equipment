import unittest
import torch
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the models to test
import sys
sys.path.append('..')
from models.simple_lstm_model import SimpleLSTM
from models.tft_model import TemporalFusionTransformer, GatedResidualNetwork, VariableSelectionNetwork
from models.tft_data_module import CMAPSSDataModule, CMAPSSDataset


class TestSimpleLSTMModel(unittest.TestCase):
    """Test cases for Simple LSTM model."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 30
        self.input_size = 24
        self.hidden_size = 32
        
        # Create dummy data
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        self.y = torch.randn(self.batch_size)
    
    def test_model_creation(self):
        """Test Simple LSTM model creation."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=0.1,
            learning_rate=0.001
        )
        
        self.assertIsInstance(model, SimpleLSTM)
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.hidden_size, self.hidden_size)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"Simple LSTM parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass through Simple LSTM."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        model.eval()
        with torch.no_grad():
            output = model(self.x)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_training_step(self):
        """Test training step."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        loss = model.training_step((self.x, self.y), 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_validation_step(self):
        """Test validation step."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        loss = model.validation_step((self.x, self.y), 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        optimizer_config = model.configure_optimizers()
        self.assertIn('optimizer', optimizer_config)
        self.assertIn('lr_scheduler', optimizer_config)


class TestTFTComponents(unittest.TestCase):
    """Test cases for TFT model components."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 30
        self.input_size = 24
        self.hidden_size = 32
    
    def test_gated_residual_network(self):
        """Test Gated Residual Network."""
        grn = GatedResidualNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=0.1
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        context = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Test without context
        output = grn(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
        # Test with context
        grn_with_context = GatedResidualNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            context_size=self.hidden_size,
            dropout=0.1
        )
        output = grn_with_context(x, context)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_variable_selection_network(self):
        """Test Variable Selection Network."""
        input_sizes = {'var1': 8, 'var2': 8, 'var3': 8}
        vsn = VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=self.hidden_size,
            dropout=0.1
        )
        
        inputs = {
            'var1': torch.randn(self.batch_size, self.seq_len, 8),
            'var2': torch.randn(self.batch_size, self.seq_len, 8),
            'var3': torch.randn(self.batch_size, self.seq_len, 8)
        }
        
        output, weights = vsn(inputs)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, 3))
        
        # Check that weights sum to 1
        weight_sums = torch.sum(weights, dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), rtol=1e-6, atol=1e-6)


class TestTFTModel(unittest.TestCase):
    """Test cases for Temporal Fusion Transformer model."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 30
        self.num_features = 24
        self.hidden_size = 32
        
        # Create dummy data
        self.x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        self.y = torch.randn(self.batch_size)
    
    def test_model_creation(self):
        """Test TFT model creation."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.num_features,
            hidden_size=self.hidden_size,
            num_lstm_layers=1,
            dropout=0.1,
            num_attention_heads=2,
            learning_rate=0.001
        )
        
        self.assertIsInstance(model, TemporalFusionTransformer)
        self.assertEqual(model.num_time_varying_real_vars, self.num_features)
        self.assertEqual(model.hidden_size, self.hidden_size)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"TFT parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass through TFT."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.num_features,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        model.eval()
        with torch.no_grad():
            output = model(self.x)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_training_step(self):
        """Test training step."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.num_features,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        loss = model.training_step((self.x, self.y), 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_validation_step(self):
        """Test validation step."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.num_features,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        loss = model.validation_step((self.x, self.y), 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.num_features,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        optimizer_config = model.configure_optimizers()
        self.assertIn('optimizer', optimizer_config)
        self.assertIn('lr_scheduler', optimizer_config)


class TestDataModule(unittest.TestCase):
    """Test cases for data module."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'test_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create dummy data files
        self.create_dummy_data()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def create_dummy_data(self):
        """Create dummy data files for testing."""
        # Create dummy numpy arrays
        X_train = np.random.randn(100, 30, 24).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(50, 30, 24).astype(np.float32)
        y_val = np.random.randn(50).astype(np.float32)
        X_test = np.random.randn(75, 30, 24).astype(np.float32)
        y_test = np.random.randn(75).astype(np.float32)
        
        # Save arrays
        np.save(os.path.join(self.data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.data_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.data_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(self.data_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(self.data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.data_dir, 'y_test.npy'), y_test)
        
        # Create metadata
        metadata = {
            'num_features': 24,
            'window_size': 30,
            'feature_columns': [f'feature_{i}' for i in range(24)]
        }
        
        import json
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
    
    def test_dataset_creation(self):
        """Test CMAPSSDataset creation."""
        X = np.random.randn(10, 30, 24)
        y = np.random.randn(10)
        
        dataset = CMAPSSDataset(X, y)
        self.assertEqual(len(dataset), 10)
        
        # Test getting an item
        x_item, y_item = dataset[0]
        self.assertEqual(x_item.shape, (30, 24))
        self.assertIsInstance(y_item, torch.Tensor)
    
    def test_data_module_creation(self):
        """Test CMAPSSDataModule creation."""
        data_module = CMAPSSDataModule(
            data_dir=self.data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=False
        )
        
        self.assertEqual(data_module.batch_size, 4)
        self.assertEqual(data_module.num_workers, 0)
    
    def test_data_module_setup(self):
        """Test data module setup."""
        data_module = CMAPSSDataModule(
            data_dir=self.data_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False
        )
        
        data_module.setup()
        
        self.assertIsNotNone(data_module.train_dataset)
        self.assertIsNotNone(data_module.val_dataset)
        self.assertIsNotNone(data_module.test_dataset)
        self.assertEqual(data_module.num_features, 24)
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        data_module = CMAPSSDataModule(
            data_dir=self.data_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False
        )
        
        data_module.setup()
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        x, y = batch
        self.assertEqual(x.shape, (4, 30, 24))
        self.assertEqual(y.shape, (4,))
        
        # Test validation dataloader
        val_loader = data_module.val_dataloader()
        batch = next(iter(val_loader))
        x, y = batch
        self.assertEqual(x.shape, (4, 30, 24))
        self.assertEqual(y.shape, (4,))
        
        # Test test dataloader
        test_loader = data_module.test_dataloader()
        batch = next(iter(test_loader))
        x, y = batch
        self.assertEqual(x.shape, (4, 30, 24))
        self.assertEqual(y.shape, (4,))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for models with data."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 30
        self.num_features = 24
        self.hidden_size = 32
        
        # Create dummy data
        self.x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        self.y = torch.randn(self.batch_size)
    
    def test_lstm_integration(self):
        """Test LSTM model integration."""
        model = SimpleLSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        # Test forward pass
        output = model(self.x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test loss calculation
        loss = torch.nn.functional.mse_loss(output.squeeze(), self.y)
        self.assertGreater(loss.item(), 0)
    
    def test_tft_integration(self):
        """Test TFT model integration."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.num_features,
            hidden_size=self.hidden_size,
            learning_rate=0.001
        )
        
        # Test forward pass
        output = model(self.x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test loss calculation
        loss = torch.nn.functional.mse_loss(output.squeeze(), self.y)
        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main() 