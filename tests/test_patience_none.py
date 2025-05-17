import numpy as np
import pytest
from pygrinder import mcar

from pypots.imputation import SAITS


def create_test_data(n_samples=100, n_steps=50, n_features=10, missing_rate=0.1):
    """Create synthetic time series data with missing values."""
    # Generate complete data
    X_ori = np.random.randn(n_samples, n_steps, n_features)
    
    # Introduce missing values
    X = mcar(X_ori, p=missing_rate)
    
    return X, X_ori

def test_patience_none_behavior():
    """Test that when patience=None, the model uses the final epoch's model."""
    # Create test data
    X, X_ori = create_test_data()
    
    # Create two identical models with different patience settings
    model_with_patience = SAITS(
        n_steps=50,
        n_features=10,
        n_layers=2,
        d_model=64,
        d_ffn=128,
        n_heads=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        epochs=10,
        patience=5,  # Early stopping enabled
        batch_size=32,
        device="cpu"
    )
    
    model_without_patience = SAITS(
        n_steps=50,
        n_features=10,
        n_layers=2,
        d_model=64,
        d_ffn=128,
        n_heads=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        epochs=10,
        patience=None,  # Early stopping disabled
        batch_size=32,
        device="cpu"
    )
    
    # Train both models
    model_with_patience.fit({"X": X})
    model_without_patience.fit({"X": X})
    
    # Get predictions from both models
    pred_with_patience = model_with_patience.predict({"X": X})
    pred_without_patience = model_without_patience.predict({"X": X})
    
    # Verify that model_with_patience uses the best model (not necessarily the last epoch)
    assert model_with_patience.best_epoch <= 10
    assert model_with_patience.best_epoch != -1
    
    # Verify that model_without_patience uses the final epoch's model
    assert model_without_patience.best_epoch == 10
    
    # Verify that the predictions are different (since they use different models)
    assert not np.array_equal(pred_with_patience["imputation"], pred_without_patience["imputation"])

def test_patience_none_no_validation_leakage():
    """Test that when patience=None, there is no validation data leakage."""
    # Create test data
    X, X_ori = create_test_data()
    
    # Split data into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    X_ori_train, X_ori_test = X_ori[:train_size], X_ori[train_size:]
    
    # Create model with patience=None
    model = SAITS(
        n_steps=50,
        n_features=10,
        n_layers=2,
        d_model=64,
        d_ffn=128,
        n_heads=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        epochs=10,
        patience=None,  # Early stopping disabled
        batch_size=32,
        device="cpu"
    )
    
    # Train model with and without test set
    model_with_test = SAITS(
        n_steps=50,
        n_features=10,
        n_layers=2,
        d_model=64,
        d_ffn=128,
        n_heads=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        epochs=10,
        patience=None,
        batch_size=32,
        device="cpu"
    )
    
    # Train both models
    model.fit({"X": X_train})
    model_with_test.fit({"X": X_train}, val_set={"X": X_test, "X_ori": X_ori_test})
    
    # Get predictions
    pred1 = model.predict({"X": X_test})
    pred2 = model_with_test.predict({"X": X_test})
    
    # Calculate MSE for both predictions
    mse1 = np.mean((pred1["imputation"] - X_ori_test) ** 2)
    mse2 = np.mean((pred2["imputation"] - X_ori_test) ** 2)
    
    # The MSEs should be similar (within a small tolerance)
    # If there was data leakage, mse2 would be significantly better than mse1
    assert abs(mse1 - mse2) < 0.1, "Significant difference in MSE suggests validation data leakage"

def test_patience_none_training_completion():
    """Test that when patience=None, training completes all epochs."""
    # Create test data
    X, _ = create_test_data()
    
    # Create model with patience=None
    model = SAITS(
        n_steps=50,
        n_features=10,
        n_layers=2,
        d_model=64,
        d_ffn=128,
        n_heads=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        epochs=10,
        patience=None,  # Early stopping disabled
        batch_size=32,
        device="cpu"
    )
    
    # Train model
    model.fit({"X": X})
    
    # Verify that training completed all epochs
    assert model.best_epoch == 10, "Training did not complete all epochs when patience=None"
    
    # Verify that the model used the final epoch's weights
    assert model.best_model_dict is not None, "No model weights were saved"
    assert model.best_loss != float("inf"), "Best loss was not updated"

if __name__ == "__main__":
    pytest.main([__file__]) 