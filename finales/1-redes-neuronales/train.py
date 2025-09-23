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
