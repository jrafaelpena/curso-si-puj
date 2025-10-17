# pso_hyperparameter_tuning.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
import optuna
import optunahub
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from sklearn.preprocessing import StandardScaler
from train import train_model
import warnings
warnings.filterwarnings('ignore')


def create_optimized_mlp(n_features, config, n_classes=2):
    """
    Create MLP with fixed 3-layer architecture and ReLU activation.
    
    Args:
        n_features: Number of input features
        config: Dictionary with hyperparameters
        n_classes: Number of output classes
    
    Returns:
        PyTorch Sequential model
    """
    layers = []
    
    # Fixed 3 hidden layers with ReLU activation
    prev_size = n_features
    
    for i in range(3):  # Fixed to 3 layers
        # Get layer size
        layer_size = config[f'hidden_size_{i}']
        
        # Linear layer
        layers.append(Linear(prev_size, layer_size))
        
        # Fixed ReLU activation
        layers.append(ReLU())
        
        # Optional batch normalization
        if config.get('use_batch_norm', False):
            layers.append(BatchNorm1d(layer_size))
        
        # Dropout (0.1 to 0.6)
        dropout_rate = config[f'dropout_rate_{i}']
        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))
        
        prev_size = layer_size
    
    # Output layer
    layers.append(Linear(prev_size, n_classes))
    
    return Sequential(*layers)


def objective_function(trial, X_train_scaled, y_train, X_val_scaled, y_val, 
                       device, epochs=200, patience=40, verbose=False):
    """
    Objective function for PSO optimization with fixed architecture constraints.
    
    Args:
        trial: Optuna trial object
        X_train_scaled, y_train: Scaled training data
        X_val_scaled, y_val: Scaled validation data
        device: torch device
        epochs: Maximum training epochs
        patience: Early stopping patience
        verbose: Print progress
    
    Returns:
        float: Validation loss (to minimize)
    """
    
    # Hyperparameters to optimize
    config = {
        # Batch normalization
        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
        
        # Adam optimizer parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        
        # Training
        'batch_size': trial.suggest_int('batch_size', 32, 128, step=16),
    }
    
    # Hidden layer sizes (3 layers, each can be different)
    # First layer: 32-256 neurons
    config['hidden_size_0'] = trial.suggest_int('hidden_size_0', 32, 256, step=32)
    
    # Second layer: 16 to at most the size of first layer
    config['hidden_size_1'] = trial.suggest_int(
        'hidden_size_1', 
        16, 
        min(config['hidden_size_0'], 192), 
        step=16
    )
    
    # Third layer: 16 to at most the size of second layer
    config['hidden_size_2'] = trial.suggest_int(
        'hidden_size_2', 
        16, 
        min(config['hidden_size_1'], 128), 
        step=16
    )
    
    # Dropout rates for each layer (0.1 to 0.6)
    config['dropout_rate_0'] = trial.suggest_float('dropout_rate_0', 0.1, 0.6)
    config['dropout_rate_1'] = trial.suggest_float('dropout_rate_1', 0.1, 0.6)
    config['dropout_rate_2'] = trial.suggest_float('dropout_rate_2', 0.1, 0.3)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'] * 2, 
        shuffle=False, 
        pin_memory=True
    )
    
    # Create model
    n_features = X_train_scaled.shape[1]
    model = create_optimized_mlp(n_features, config)
    
    # Loss and optimizer (fixed to Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    try:
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            patience=patience,
            silent=not verbose
        )
        
        val_loss = results['best_val_metrics']['loss']
        val_acc = results['best_val_metrics']['accuracy']
        
        # Report intermediate value for pruning
        trial.report(val_loss, results['total_epochs'])
        
        # Prune if trial is not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if verbose:
            print(f"Trial {trial.number}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        return val_loss
        
    except Exception as e:
        if verbose:
            print(f"Trial {trial.number} failed: {e}")
        return float('inf')


def setup_pso_search_space(n_features=11):
    """
    Setup the search space for PSO optimization with fixed constraints.
    
    Args:
        n_features: Number of input features
    
    Returns:
        dict: Search space dictionary for PSO sampler
    """
    search_space = {
        # Batch normalization
        'use_batch_norm': CategoricalDistribution([True, False]),
        
        # Adam optimizer parameters
        'learning_rate': FloatDistribution(1e-5, 1e-2, log=True),
        'weight_decay': FloatDistribution(1e-6, 1e-2, log=True),
        
        # Training
        'batch_size': IntDistribution(16, 128, step=16),
        
        # Layer sizes (3 fijas)
        'hidden_size_0': IntDistribution(32, 256, step=32),
        'hidden_size_1': IntDistribution(16, 192, step=16),
        'hidden_size_2': IntDistribution(16, 128, step=16),
        
        # Dropout rates
        'dropout_rate_0': FloatDistribution(0.1, 0.6),
        'dropout_rate_1': FloatDistribution(0.1, 0.6),
        'dropout_rate_2': FloatDistribution(0.1, 0.6),
    }
    
    return search_space


def run_pso_hyperparameter_tuning(X_train_scaled, y_train, X_val_scaled, y_val, 
                                  device, n_trials=100, n_particles=20,
                                  epochs_per_trial=200, patience=40,
                                  inertia=0.7, cognitive=1.4, social=1.4,
                                  verbose=True, seed=42):
    """
    Run PSO hyperparameter optimization with fixed 3-layer ReLU architecture and Adam optimizer.
    
    Args:
        X_train_scaled, y_train: Scaled training data (11 features)
        X_val_scaled, y_val: Scaled validation data (11 features)
        device: torch device
        n_trials: Total number of trials
        n_particles: Number of particles in PSO swarm
        epochs_per_trial: Max epochs for each trial
        patience: Early stopping patience
        inertia: PSO inertia weight
        cognitive: PSO cognitive coefficient
        social: PSO social coefficient
        verbose: Print progress
        seed: Random seed
    
    Returns:
        dict: Results including best parameters and study object
    """
    
    # Validate input dimensions
    n_features = X_train_scaled.shape[1]
    if n_features != 11:
        print(f"Warning: Expected 11 features but got {n_features}")
    
    if verbose:
        print(f"Starting PSO Hyperparameter Optimization")
        print(f"Fixed Architecture: 3 hidden layers, ReLU activation, Adam optimizer")
        print(f"Input features: {n_features}")
        print(f"Training samples: {X_train_scaled.shape[0]}")
        print(f"Validation samples: {X_val_scaled.shape[0]}")
        print(f"PSO Settings: {n_particles} particles, {n_trials} trials")
        print(f"PSO Coefficients: inertia={inertia}, cognitive={cognitive}, social={social}")
        print("=" * 80)
    
    # Setup search space
    search_space = setup_pso_search_space(n_features)
    
    # Load and configure PSO sampler
    pso_module = optunahub.load_module(package="samplers/pso")
    sampler = pso_module.PSOSampler(
        search_space=search_space,
        n_particles=n_particles,
        inertia=inertia,
        cognitive=cognitive,
        social=social,
        seed=seed
    )
    
    # Create study with pruning
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=50,
            interval_steps=10
        )
    )
    
    # Define objective with fixed parameters
    def wrapped_objective(trial):
        return objective_function(
            trial, X_train_scaled, y_train, X_val_scaled, y_val,
            device, epochs_per_trial, patience, verbose=False
        )
    
    # Callback for progress tracking
    def callback(study, trial):
        if verbose and trial.number % 10 == 0:
            print(f"Trial {trial.number}/{n_trials} | "
                  f"Best value: {study.best_value:.4f} | "
                  f"Best trial: {study.best_trial.number}")
    
    # Run optimization
    study.optimize(
        wrapped_objective,
        n_trials=n_trials,
        callbacks=[callback] if verbose else None,
        show_progress_bar=verbose
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"PSO Optimization Complete!")
        print(f"Best validation loss: {best_value:.4f}")
        print(f"Best trial: {study.best_trial.number}")
        print("\nBest Hyperparameters:")
        print(f"  Architecture: 3 layers with ReLU activation")
        print(f"  Layer 1: {best_params['hidden_size_0']} neurons, dropout={best_params['dropout_rate_0']:.3f}")
        print(f"  Layer 2: {best_params['hidden_size_1']} neurons, dropout={best_params['dropout_rate_1']:.3f}")
        print(f"  Layer 3: {best_params['hidden_size_2']} neurons, dropout={best_params['dropout_rate_2']:.3f}")
        print(f"  Learning rate: {best_params['learning_rate']:.6f}")
        print(f"  Weight decay: {best_params['weight_decay']:.6f}")
        print(f"  Batch size: {best_params['batch_size']}")
        print(f"  Batch normalization: {best_params['use_batch_norm']}")
    
    # Create final model with best parameters
    final_config = best_params.copy()
    final_model = create_optimized_mlp(n_features, final_config)
    
    # Train final model with best parameters
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=final_config.get('batch_size', 64),
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=final_config.get('batch_size', 64) * 2,
        shuffle=False,
        pin_memory=True
    )
    
    # Setup Adam optimizer for final model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=final_config['learning_rate'],
        weight_decay=final_config['weight_decay']
    )
    
    # Train final model
    if verbose:
        print("\nTraining final model with best hyperparameters...")
    
    final_results = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=300,  # More epochs for final training
        patience=50,
        silent=not verbose
    )
    
    if verbose:
        print(f"\nFinal Model Performance:")
        print(f"  Training Loss: {final_results['best_train_metrics']['loss']:.4f}")
        print(f"  Training Acc: {final_results['best_train_metrics']['accuracy']:.4f}")
        print(f"  Validation Loss: {final_results['best_val_metrics']['loss']:.4f}")
        print(f"  Validation Acc: {final_results['best_val_metrics']['accuracy']:.4f}")
    
    return {
        'study': study,
        'best_params': best_params,
        'best_value': best_value,
        'best_trial': study.best_trial,
        'final_model': final_model,
        'final_results': final_results,
        'n_trials': n_trials,
        'n_features': n_features
    }


def print_model_summary(model, n_features=11):
    """
    Print a summary of the model architecture.
    
    Args:
        model: The PyTorch model
        n_features: Number of input features
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Architecture Summary:")
    print("=" * 60)
    print(f"Input features: {n_features}")
    print("\nLayers:")
    for i, layer in enumerate(model):
        print(f"  {i}: {layer}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)


# Example usage function
def example_usage():
    """
    Example of how to use the PSO hyperparameter tuning with fixed architecture.
    """
    # Assuming you have your data ready (11 features, already scaled)
    # X_train_scaled: shape (n_train_samples, 11)
    # y_train: shape (n_train_samples,)
    # X_val_scaled: shape (n_val_samples, 11)
    # y_val: shape (n_val_samples,)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run PSO hyperparameter tuning
    pso_results = run_pso_hyperparameter_tuning(
        X_train_scaled=X_train_scaled,  # Your scaled training data (11 features)
        y_train=y_train,
        X_val_scaled=X_val_scaled,      # Your scaled validation data (11 features)
        y_val=y_val,
        device=device,
        n_trials=100,        # Total number of trials
        n_particles=20,      # Number of particles in swarm
        epochs_per_trial=200,  # Max epochs per trial
        patience=40,         # Early stopping patience
        inertia=0.7,        # PSO inertia weight
        cognitive=1.4,      # Cognitive coefficient
        social=1.4,         # Social coefficient
        verbose=True,
        seed=42
    )
    
    # Access results
    best_model = pso_results['final_model']
    best_params = pso_results['best_params']
    study = pso_results['study']
    
    # Print model summary
    print_model_summary(best_model, n_features=11)
    
    # Visualize optimization history (optional)
    try:
        import plotly.graph_objects as go
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.show()
        
        # Plot parameter importance
        fig = plot_param_importances(study)
        fig.show()
    except ImportError:
        print("Install plotly for visualization: pip install plotly")
    
    return pso_results


if __name__ == "__main__":
    # This will only run if you execute this file directly
    print("PSO Hyperparameter Tuning Module - Fixed Architecture Version")
    print("Architecture: 3 hidden layers with ReLU activation and Adam optimizer")
    print("Use run_pso_hyperparameter_tuning() function to optimize your model")
    
    # Uncomment to run example (requires your data)
    # pso_results = example_usage()
