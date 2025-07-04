import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the functions to test
import sys
sys.path.append('..')
from preprocess_data import load_dataset, calculate_rul, normalize_data, prepare_cmapss_data


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create sample test data
        self.create_sample_data()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_data(self):
        """Create sample CMAPSS-like data for testing."""
        # Sample training data
        train_data = []
        for unit in range(1, 4):  # 3 units
            for cycle in range(1, 21):  # 20 cycles each
                row = [unit, cycle, 0.5, 0.3, 0.2]  # op_settings
                row.extend([0.1 * i for i in range(1, 22)])  # sensor readings
                train_data.append(row)
        
        train_df = pd.DataFrame(train_data, columns=[
            'unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 22)])
        
        # Sample test data
        test_data = []
        for unit in range(1, 3):  # 2 units
            for cycle in range(1, 16):  # 15 cycles each
                row = [unit, cycle, 0.5, 0.3, 0.2]  # op_settings
                row.extend([0.1 * i for i in range(1, 22)])  # sensor readings
                test_data.append(row)
        
        test_df = pd.DataFrame(test_data, columns=[
            'unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 22)])
        
        # Sample RUL data
        rul_data = [[10], [15]]  # RUL values for 2 test units
        rul_df = pd.DataFrame(rul_data, columns=['RUL'])
        
        # Save test files
        train_df.to_csv(os.path.join(self.data_dir, 'train_FD001.txt'), 
                       sep=' ', header=False, index=False)
        test_df.to_csv(os.path.join(self.data_dir, 'test_FD001.txt'), 
                      sep=' ', header=False, index=False)
        rul_df.to_csv(os.path.join(self.data_dir, 'RUL_FD001.txt'), 
                     sep=' ', header=False, index=False)
    
    def test_load_dataset(self):
        """Test loading dataset function."""
        train_df, test_df, rul_df = load_dataset('FD001', data_dir=self.data_dir)
        
        # Check that data is loaded correctly
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertIsInstance(rul_df, pd.DataFrame)
        
        # Check column names
        expected_cols = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        self.assertEqual(list(train_df.columns), expected_cols)
        self.assertEqual(list(test_df.columns), expected_cols)
        self.assertEqual(list(rul_df.columns), ['RUL'])
        
        # Check data shapes
        self.assertEqual(train_df.shape[0], 60)  # 3 units * 20 cycles
        self.assertEqual(test_df.shape[0], 30)   # 2 units * 15 cycles
        self.assertEqual(rul_df.shape[0], 2)     # 2 test units
    
    def test_calculate_rul(self):
        """Test RUL calculation function."""
        # Create test data
        test_data = []
        for unit in range(1, 3):
            for cycle in range(1, 11):
                test_data.append([unit, cycle, 0.1, 0.2, 0.3])
        
        df = pd.DataFrame(test_data, columns=['unit', 'cycle', 'sensor_1', 'sensor_2', 'sensor_3'])
        
        # Calculate RUL
        result_df = calculate_rul(df)
        
        # Check that RUL column is added
        self.assertIn('RUL', result_df.columns)
        
        # Check RUL calculations
        unit1_rul = result_df[result_df['unit'] == 1]['RUL'].values
        unit2_rul = result_df[result_df['unit'] == 2]['RUL'].values
        
        # Unit 1: max cycle is 10, so RUL should be [9, 8, 7, ..., 0]
        expected_unit1 = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        np.testing.assert_array_equal(unit1_rul, expected_unit1)
        
        # Unit 2: max cycle is 10, so RUL should be [9, 8, 7, ..., 0]
        expected_unit2 = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        np.testing.assert_array_equal(unit2_rul, expected_unit2)
    
    def test_normalize_data(self):
        """Test data normalization function."""
        # Create test data
        train_data = [[1, 0.1, 0.2, 0.3], [2, 0.4, 0.5, 0.6], [3, 0.7, 0.8, 0.9]]
        test_data = [[4, 0.2, 0.3, 0.4], [5, 0.5, 0.6, 0.7]]
        
        train_df = pd.DataFrame(train_data, columns=['unit', 'sensor_1', 'sensor_2', 'sensor_3'])
        test_df = pd.DataFrame(test_data, columns=['unit', 'sensor_1', 'sensor_2', 'sensor_3'])
        
        # Normalize data
        norm_train_df, norm_test_df, scaler = normalize_data(train_df, test_df)
        
        # Check that data is normalized
        self.assertIsInstance(norm_train_df, pd.DataFrame)
        self.assertIsInstance(norm_test_df, pd.DataFrame)
        self.assertIsNotNone(scaler)
        
        # Check that sensor columns are normalized (values between 0 and 1)
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3']
        for col in sensor_cols:
            self.assertTrue((norm_train_df[col] >= 0).all())
            self.assertTrue((norm_train_df[col] <= 1).all())
            self.assertTrue((norm_test_df[col] >= 0).all())
            self.assertTrue((norm_test_df[col] <= 1).all())
    
    def test_prepare_cmapss_data(self):
        """Test the main data preparation function."""
        # Test with a single dataset
        result = prepare_cmapss_data(dataset_ids=['FD001'], val_size=0.3)
        
        # Check return structure
        self.assertIn('train', result)
        self.assertIn('val', result)
        self.assertIn('test', result)
        self.assertIn('scaler', result)
        
        # Check data types
        self.assertIsInstance(result['train'], pd.DataFrame)
        self.assertIsInstance(result['val'], pd.DataFrame)
        self.assertIsInstance(result['test'], pd.DataFrame)
        self.assertIsNotNone(result['scaler'])
        
        # Check that RUL column exists in all datasets
        self.assertIn('RUL', result['train'].columns)
        self.assertIn('RUL', result['val'].columns)
        self.assertIn('RUL', result['test'].columns)
        
        # Check that no missing values exist
        self.assertEqual(result['train'].isnull().sum().sum(), 0)
        self.assertEqual(result['val'].isnull().sum().sum(), 0)
        self.assertEqual(result['test'].isnull().sum().sum(), 0)
    
    def test_invalid_dataset_id(self):
        """Test handling of invalid dataset ID."""
        with self.assertRaises(FileNotFoundError):
            load_dataset('INVALID', data_dir=self.data_dir)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['unit', 'cycle', 'sensor_1'])
        # Empty DataFrame should not raise ValueError, just return empty result
        result = calculate_rul(empty_df)
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main() 