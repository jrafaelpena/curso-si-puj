# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, ReLU, Dropout
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def _compute_metrics(y_true, y_pred, y_probs):
    """Compute binary classification metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    try:
        auroc = roc_auc_score(y_true, y_probs[:, 1])
    except ValueError as e:
        print(f"AUROC computation failed: {e}")
        auroc = 0.0
    return precision, recall, f1, auroc

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        # Reiniciar los gradientes antes del feed forward
        optimizer.zero_grad()
        
        # Feed forward (Predición)
        output = model(features)
        
        # Cálculo de pérdida
        loss = criterion(output, labels)

        # Backpropagation
        loss.backward()
        
        # Gradient Descent (Se optimizan los parámetros)
        optimizer.step()
        
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(output, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())
        all_probs.extend(F.softmax(output, dim=1).cpu().detach().numpy())
    
    loss = running_loss / total
    acc = 100.0 * correct / total
    
    return loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            output = model(features)
            loss = criterion(output, labels)
            
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(output, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(F.softmax(output, dim=1).cpu().detach().numpy())
    
    loss = running_loss / total
    acc = 100.0 * correct / total

    return loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=300, patience=75, print_every=10, save_path='best_model.pth',
                silent=False):
    """
    Complete training loop with early stopping and metric tracking.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device (cuda/cpu)
        epochs: Maximum number of epochs
        patience: Early stopping patience
        print_every: Print metrics every N epochs
        save_path: Path to save best model
        silent: If True, suppress all print statements
    
    Returns:
        dict: Training results containing history and best metrics
    """
    model.to(device)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # History tracking
    history_loss_train = []
    history_loss_val = []
    history_acc_train = []
    history_acc_val = []
    
    # Best model metrics storage
    best_train_metrics = {}
    best_val_metrics = {}
    
    if not silent:
        print(f"Starting training for {epochs} epochs with patience={patience}")
        print("=" * 100)
    
    for epoch in range(epochs):
        # Train and validate
        train_loss, train_acc, train_labels, train_preds, train_probs = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_labels, val_preds, val_probs = eval_epoch(
            model, val_loader, criterion, device
        )
        
        # Store loss and accuracy history
        history_loss_train.append(train_loss)
        history_loss_val.append(val_loss)
        history_acc_train.append(train_acc)
        history_acc_val.append(val_acc)
        
        # Compute detailed metrics
        train_precision, train_recall, train_f1, train_auroc = _compute_metrics(
            train_labels, train_preds, train_probs
        )
        val_precision, val_recall, val_f1, val_auroc = _compute_metrics(
            val_labels, val_preds, val_probs
        )
        
        # Print metrics
        if not silent and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: Loss {train_loss:.4f} Acc {train_acc:.2f}% F1 {train_f1:.4f} AUROC {train_auroc:.4f} | "
                  f"Val: Loss {val_loss:.4f} Acc {val_acc:.2f}% F1 {val_f1:.4f} AUROC {val_auroc:.4f}")
        
        # Early stopping and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), save_path)
            
            # Store best model metrics
            best_train_metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1,
                'auroc': train_auroc
            }
            
            best_val_metrics = {
                'loss': val_loss,
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'auroc': val_auroc
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if not silent:
                    print("=" * 100)
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if not silent:
        print("=" * 100)
        print(f"Training completed. Best model from epoch {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Return comprehensive results
    return {
        'history': {
            'train_loss': history_loss_train,
            'val_loss': history_loss_val,
            'train_acc': history_acc_train,
            'val_acc': history_acc_val
        },
        'best_train_metrics': best_train_metrics,
        'best_val_metrics': best_val_metrics,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'model': model
    }