# genetic_feature_selection.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, ReLU, Dropout
from deap import base, creator, tools, algorithms
import random
from sklearn.preprocessing import StandardScaler
from train import train_model


def create_mlp(n_features, hidden_sizes=[64, 32, 16], dropout_rates=[0.4, 0.4, 0.1], n_classes=2):
    """
    Create MLP with dynamic input size based on selected features.
    
    Args:
        n_features: Number of input features
        hidden_sizes: List of hidden layer sizes
        dropout_rates: List of dropout rates for each layer
        n_classes: Number of output classes
    
    Returns:
        PyTorch Sequential model
    """
    layers = []
    
    # Input layer
    layers.extend([
        Linear(n_features, hidden_sizes[0]),
        ReLU(),
        Dropout(dropout_rates[0])
    ])
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.extend([
            Linear(hidden_sizes[i], hidden_sizes[i+1]),
            ReLU(),
            Dropout(dropout_rates[i+1])
        ])
    
    # Output layer
    layers.append(Linear(hidden_sizes[-1], n_classes))
    
    return Sequential(*layers)


def evaluate_individual(individual, X_train, y_train, X_val, y_val, 
                       scaler, device, epochs=100, patience=25, verbose=False):
    """
    Evaluate a feature subset (individual) by training an MLP.
    
    Args:
        individual: Binary list indicating selected features
        X_train, y_train: Training data
        X_val, y_val: Validation data
        scaler: StandardScaler instance
        device: torch device
        epochs: Max training epochs
        patience: Early stopping patience
        verbose: Print individual evaluation results
    
    Returns:
        Tuple with fitness score (negative validation loss, negative number of features)
    """
    # Convert individual to feature indices
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    
    # At least one feature must be selected
    if len(selected_features) == 0:
        if verbose:
            print(f"Individual with 0 features selected - Invalid")
        return (float('-inf'), -len(individual))  # Worst possible score for minimization
    
    # Select features
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    # Create model
    model = create_mlp(n_features=len(selected_features))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    
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
            silent=True
        )
        
        # Get validation loss
        val_loss = results['best_val_metrics']['loss']
        fitness = (-val_loss, -len(selected_features))
        
        if verbose:
            print(f"Individual finished: {len(selected_features)} features | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Fitness: {fitness[0]:.4f} | "
                  f"Epochs: {results['total_epochs']}")
        
        # Return fitness: (negative loss for maximization, negative number of features for parsimony)
        # DEAP maximizes, so we negate the loss to minimize it
        return fitness
    
    except Exception as e:
        if verbose:
            print(f"ERROR: Individual with {len(selected_features)} features failed: {e}")
        return (float('-inf'), -len(selected_features))


def setup_deap(n_features, weights=(1.0, 0.01)):
    """
    Setup DEAP framework for genetic algorithm.
    
    Args:
        n_features: Total number of features
        weights: Weights for multi-objective optimization (loss, parsimony)
    
    Returns:
        toolbox: DEAP toolbox
    """
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # binario (0 - 1)
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Operadores Genéticos
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox


def run_genetic_algorithm(X_train, y_train, X_val, y_val, device,
                         n_generations=20, population_size=30,
                         cx_prob=0.7, mut_prob=0.2,
                         epochs_per_individual=100, patience=25,
                         verbose=True, verbose_individuals=True):
    """
    Run genetic algorithm for feature selection.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        device: torch device
        n_generations: Number of generations
        population_size: Size of population
        cx_prob: Crossover probability
        mut_prob: Mutation probability
        epochs_per_individual: Epochs to train each model
        patience: Early stopping patience
        verbose: Print generation-level progress
        verbose_individuals: Print individual-level progress
    
    Returns:
        dict: Results including best individual, fitness history, and selected features
    """
    n_features = X_train.shape[1]
    
    # Setup DEAP
    toolbox = setup_deap(n_features)
    
    # Register evaluation function
    scaler = StandardScaler()
    toolbox.register("evaluate", evaluate_individual,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    scaler=scaler, device=device,
                    epochs=epochs_per_individual, patience=patience,
                    verbose=verbose_individuals)

    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Hall of fame to keep track of best individuals
    hof = tools.HallOfFame(5)
    
    if verbose:
        print(f"Starting genetic algorithm with {n_features} features")
        print(f"Population size: {population_size}, Generations: {n_generations}")
        print("=" * 100)
    
    # Run genetic algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose
    )
    
    # Get best individual
    best_individual = hof[0]
    best_features = [i for i, bit in enumerate(best_individual) if bit == 1]
    
    if verbose:
        print("=" * 100)
        print(f"Best individual fitness: {best_individual.fitness.values}")
        print(f"Selected {len(best_features)} features: {best_features}")
    
    return {
        'best_individual': best_individual,
        'best_features': best_features,
        'best_fitness': best_individual.fitness.values,
        'hall_of_fame': hof,
        'population': population,
        'logbook': logbook,
        'n_selected_features': len(best_features)
    }
