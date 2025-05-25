import argparse
import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import sys
import logging
import utils

from extract_features import to_feature_filename
sys.path.append(os.getcwd())
import json
import numpy as np
from torch.utils.data import Dataset

from classifier import LinearClassifier
from datetime import datetime
from dataset import MMIMDBFeaturesDatasetSup, NUM_GENRES
from sklearn.metrics import accuracy_score
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy, F1Score, AUROC
from constants import DATA_SPLITS, ROOT, SAMPLE_FORMAT, get_sampled_save_path
from models import load_llm, load_tokenizer
from tasks import get_models

# Function to set up logging
def setup_logging(log_dir="logs"):
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a new log file for each experiment with a timestamp
    log_filename = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    # Set up logging configuration
    logging.basicConfig(
        filename=log_filename,  # Unique log file for each run
        level=logging.INFO,  # Log level (INFO logs everything)
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
    )

    logging.info("Logging started for this experiment.")

def train_linear_classifier_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    for features, labels in train_loader:
        
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(features)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(outputs)
        predictions = (outputs > 0).long() 
        all_predictions.append(predictions)
        all_labels.append(labels)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    train_metrics = evaluate_predictions(all_predictions, all_labels, device)
    return model, total_loss / len(all_predictions), train_metrics["f1_mean"]

def evaluate_performance(model, data_loader, criterion, device):
    criterion = criterion.to(device)
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item() * len(outputs)
            predictions = (outputs > 0).long() 
            all_predictions.append(predictions)
            all_labels.append(labels)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    return total_loss / len(all_predictions), evaluate_predictions(all_predictions, all_labels, device)



def train_linear_classifier(input_dim, train_loader, val_loader, criterion, device, model_save_dir, 
                            lr=1e-5, weight_decay=0.0, num_epochs=10):
    # Dataset and DataLoader
    output_dim = NUM_GENRES  # Number of genres

    classifier_model = LinearClassifier(input_dim, output_dim).to(device)

    # Loss function and optimizer
    criterion = criterion.to(device)
    optimizer = optim.Adam(classifier_model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -np.inf
    best_val_metrics = None
    save_path = os.path.join(model_save_dir, f"lr_{lr}_wd_{weight_decay}_classifier.pt")
    final_save_path = os.path.join(model_save_dir, f"lr_{lr}_wd_{weight_decay}_final_classifier.pt")
    # Train the model
    if not os.path.exists(final_save_path):
        for epoch in range(num_epochs):  # Number of epochs
            classifier_model, loss, f1_mean = train_linear_classifier_epoch(classifier_model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = evaluate_performance(classifier_model, val_loader, criterion, device)
            val_f1_mean = val_metrics["f1_mean"]
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, F1: {f1_mean:.2f}, Val Loss: {val_loss:.4f}, F1: {val_f1_mean:.2f}")
            if val_f1_mean > best_val:
                best_val = val_f1_mean
                best_val_metrics = val_metrics
                logging.info("Saving best model")
                torch.save(classifier_model, save_path)
        best_model = torch.load(save_path, weights_only=False)
        logging.info(f"Saving final model to: {final_save_path}")
        torch.save(best_model, final_save_path)
    else:
        logging.info(f"Loading final model from: {final_save_path}")
        best_model = torch.load(final_save_path, weights_only=False)
        _, best_val_metrics = evaluate_performance(best_model, val_loader, criterion, device)
    return best_model, best_val_metrics

def tune_hyperparameters(input_dim, train_loader, val_loader, criterion, device, model_save_path, 
                         lr_values=[1e-5, 1e-4, 1e-3], weight_decay_values=[0.0, 1e-4, 1e-3], num_epochs=10):
    # Log experiment setup
    logging.info(f"Starting hyperparameter tuning with the following parameters:")
    logging.info(f"Learning rates: {lr_values}")
    logging.info(f"Weight decays: {weight_decay_values}")
    logging.info(f"Epochs: {num_epochs}")

    best_lr = None
    best_weight_decay = None
    best_f1 = -np.inf
    best_model = None
    curr_save_dir = os.path.join(os.path.dirname(model_save_path), "temp")
    os.makedirs(curr_save_dir, exist_ok=True)
    # Grid search over learning rates and weight decays
    for lr in lr_values:
        for weight_decay in weight_decay_values:
            logging.info(f"Training with lr={lr}, weight_decay={weight_decay}")
            # curr_save_path = os.path.join(os.path.dirname(model_save_path), "temp", "classifier.pt")
            # Call the existing function to train and evaluate the model
            curr_model, val_metrics = train_linear_classifier(
                input_dim=input_dim, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                criterion=criterion, 
                device=device, 
                model_save_dir=curr_save_dir,
                lr=lr,
                weight_decay=weight_decay,
                num_epochs=num_epochs
            )
            # Assuming val_score is a dictionary with "f1_mean" as one of the keys
            logging.info(f"Validation F1 score: {val_metrics['f1_mean']:.2f}")
            if val_metrics["f1_mean"] > best_f1:
                best_f1 = val_metrics["f1_mean"]
                best_lr = lr
                best_weight_decay = weight_decay
                # Reload the best model after training
                best_model = curr_model
                # Log the best hyperparameters
                logging.info(f"New best F1 score: {best_f1:.2f} (lr={best_lr}, weight_decay={best_weight_decay})")

    logging.info(f"Best hyperparameters found - Learning Rate: {best_lr}, Weight Decay: {best_weight_decay}, Best F1: {best_f1:.2f}")
    # Save the best model
    torch.save(best_model, model_save_path)
    logging.info(f"Best model saved at {model_save_path}")


def evaluate_predictions(predictions, labels, device, use_mean_accuracy=True):
    num_labels = labels.shape[-1]
    average = "macro" if use_mean_accuracy else "micro"
    acc1 = MultilabelAccuracy(num_labels=num_labels, average=average).to(device)
    f1_score_class = F1Score(task="multilabel", num_labels=num_labels, average=None).to(device)
    f1_weighted_score = F1Score(task="multilabel", num_labels=num_labels, average="weighted").to(device)

    accuracy1 = float(acc1(predictions, labels))
    f1_score_per_class = f1_score_class(predictions, labels).cpu().numpy()
    f1_mean = float(f1_score_per_class.mean())
    f1_weighted = float(f1_weighted_score(predictions, labels))
    acc1_sklearn = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    # Compute the individual class F1 scores
    return {"acc1": accuracy1, "f1_mean": f1_mean, "f1_per_class": f1_score_per_class, "f1_weighted": f1_weighted, "acc1(subset)": acc1_sklearn}

def compute_pos_weights_multilabel(train_labels):
    # Convert train_labels to a tensor for easier manipulation
    train_labels = np.array(train_labels)
    # Count the number of positive labels for each class (column-wise sum)
    label_sums = train_labels.sum(axis=0)  # Sum over samples for each label
    
    # Compute the inverse of class frequency as the weight for the positive class
    total_samples = train_labels.shape[0]
    pos_weight = (total_samples - label_sums) / (label_sums + 1e-6)  # Add a small epsilon to avoid division by zero
    pos_weight = torch.tensor(pos_weight, dtype=torch.float)
    
    return pos_weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--pool", type=str, default='last', choices=['last', 'cls'])
    parser.add_argument("--lr_values", nargs='+', type=float, help="learning rates", default=[1e-4, 5e-4, 1e-3, 5e-3])
    parser.add_argument("--weight_decay_values", nargs='+', type=float, help="weight decays", default=[0, 0.1, 1e-2, 1e-3, 1e-4])
    parser.add_argument("--layer_idx", type=int, default=-1, help="Which layer to train the classifier on top of")
    parser.add_argument("--input_dir", type=str, default="/scratch/platonic/mmimdb/mmimdb_feat")
    parser.add_argument("--save_dir", type=str, default="/scratch/platonic/mmimdb/mmimdb_performance_upd")
    args = parser.parse_args()
    # Example of how you would call the function
    setup_logging(log_dir="logs")  # This will create a new log file in the "logs" directory
    device = "cuda:0"
    llm_models, lvm_models = get_models(args.modelset, modality="all")
    data_features = {model_name: 
                     {data_split: to_feature_filename(args.input_dir, model_name, data_split) for data_split in DATA_SPLITS} 
                     for model_name in llm_models}

    # Extract features
    for model_name in llm_models:
        # Train the model with validation
        train_dataset = MMIMDBFeaturesDatasetSup(data_features[model_name]["train"], ROOT, split="train", 
                                                 layer_idx=args.layer_idx, sampled_ids=SAMPLE_FORMAT.format(split="train"))
        val_dataset = MMIMDBFeaturesDatasetSup(data_features[model_name]["dev"], ROOT, split="dev", 
                                               layer_idx=args.layer_idx, sampled_ids=SAMPLE_FORMAT.format(split="dev"))  # Validation split
        test_dataset = MMIMDBFeaturesDatasetSup(data_features[model_name]["test"], ROOT, split="test", 
                                                layer_idx=args.layer_idx, sampled_ids=SAMPLE_FORMAT.format(split="test"))  # Test split
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        pos_weight = compute_pos_weights_multilabel(train_dataset.get_labels())
        # Weight each class to deal with class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        input_dim = train_dataset[0][0].shape[0]
        model_save_dir = utils.to_model_savedir(args.save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, "best_classifier.pt")
        if not os.path.exists(model_save_path):
            tune_hyperparameters(input_dim, train_loader, val_loader, criterion, device, model_save_path,
                                lr_values=args.lr_values, weight_decay_values=args.weight_decay_values, num_epochs=args.num_epochs)
        logging.info(f"Loading model from: {model_save_path}")
        model = torch.load(model_save_path, map_location="cpu", weights_only=False).to(device)
        test_loss, test_metrics = evaluate_performance(model, test_loader, criterion, device)
        # Save test performance
        perf_save_path = os.path.join(model_save_dir, "test_perf.npy")
        logging.info(f"Saving test metrics to: {perf_save_path}")
        np.save(perf_save_path, test_metrics)