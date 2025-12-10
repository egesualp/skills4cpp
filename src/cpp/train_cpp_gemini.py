import argparse
import os
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP for Career Path Prediction with Optuna.")
    
    parser.add_argument(
        "--features_dir", 
        type=str, 
        default="precomputed_features/decorte_full_descr_mean/",
        help="Directory containing the feature .npy files."
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=50,
        help="Number of Optuna trials to run."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30,
        help="Max epochs to train each trial."
    )
    parser.add_argument(
        "--use_text", 
        action='store_true', 
        default=True,
        help="Use the '..._text.npy' feature."
    )
    parser.add_argument(
        "--use_skill_text", 
        action='store_true', 
        help="Use the '..._skill_text.npy' feature."
    )
    # This will find ALL files starting with _structured_ and add them
    parser.add_argument(
        "--use_structured", 
        action='store_true', 
        help="Use all '..._structured_....npy' features."
    )
    
    return parser.parse_args()

# --- Helper Functions ---

def load_data(features_dir, use_text, use_skill_text, use_structured):
    """Loads all specified feature and label arrays from disk."""
    
    print("Loading data...")
    X_train_list, X_val_list, X_test_list = [], [], []
    
    # --- Load Text Modality ---
    if use_text:
        print("  > Loading text features...")
        X_train_list.append(np.load(os.path.join(features_dir, "train_text.npy")))
        X_val_list.append(np.load(os.path.join(features_dir, "val_text.npy")))
        X_test_list.append(np.load(os.path.join(features_dir, "test_text.npy")))
        
    # --- Load Skill-Text Modality ---
    if use_skill_text:
        print("  > Loading skill-text features...")
        X_train_list.append(np.load(os.path.join(features_dir, "train_skill_text.npy")))
        X_val_list.append(np.load(os.path.join(features_dir, "val_skill_text.npy")))
        X_test_list.append(np.load(os.path.join(features_dir, "test_skill_text.npy")))
        
    # --- Load All Structured Modalities ---
    if use_structured:
        print("  > Loading all structured features...")
        for f in os.listdir(features_dir):
            if f.startswith("train_structured_"):
                key = f.replace("train_structured_", "").replace(".npy", "")
                print(f"    - Loading structured feature: {key}")
                X_train_list.append(np.load(os.path.join(features_dir, f"train_structured_{key}.npy")))
                X_val_list.append(np.load(os.path.join(features_dir, f"val_structured_{key}.npy")))
                X_test_list.append(np.load(os.path.join(features_dir, f"test_structured_{key}.npy")))

    if not X_train_list:
        raise ValueError("No features selected! Please use at least one --use_... flag.")

    # Concatenate all selected features
    X_train = np.concatenate(X_train_list, axis=1)
    X_val = np.concatenate(X_val_list, axis=1)
    X_test = np.concatenate(X_test_list, axis=1)
    
    # Load labels
    Y_train = np.load(os.path.join(features_dir, "train_y.npy"))
    Y_val = np.load(os.path.join(features_dir, "val_y.npy"))
    Y_test = np.load(os.path.join(features_dir, "test_y.npy"))
    
    # Load all possible target vectors for evaluation
    Y_target_all = np.load(os.path.join(features_dir, "train_y.npy")) # Just need a full set
    Y_target_all = np.unique(Y_target_all, axis=0)
    
    print(f"Input feature dimension: {X_train.shape[1]}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_target_all

def calculate_mrr(y_pred_vectors, y_true_vectors, Y_target_all):
    """Calculates Mean Reciprocal Rank (MRR) for vector predictions."""
    
    # Calculate cosine similarity between all predictions and all possible targets
    # Shape: (n_samples, n_all_targets)
    sim_matrix = cosine_similarity(y_pred_vectors, Y_target_all)
    
    # Get the indices that would sort this matrix, in descending order
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    
    # Get the true target vectors
    true_target_indices = []
    for y_true in y_true_vectors:
        # Find the index of the true target in the Y_target_all array
        true_index = np.where((Y_target_all == y_true).all(axis=1))[0][0]
        true_target_indices.append(true_index)
    
    reciprocal_ranks = []
    for i in range(len(y_pred_vectors)):
        true_idx = true_target_indices[i]
        # Find the rank of the true target in our sorted list
        rank_list = list(sorted_indices[i])
        rank = rank_list.index(true_idx) + 1
        reciprocal_ranks.append(1.0 / rank)
        
    return np.mean(reciprocal_ranks)

# --- PyTorch Model Definition ---

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, hidden_dim, dropout_rate):
        super(MLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Optuna Objective Function ---

def objective(trial, X_train, Y_train, X_val, Y_val, Y_target_all, args):
    """
    Optuna objective function to train and evaluate a single model.
    """
    # 1. Suggest Hyperparameters
    # We ask Optuna to suggest values for us to try
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # 2. Setup DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
    
    # 3. Build Model, Loss, Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    model = MLP(input_dim, output_dim, n_layers, hidden_dim, dropout_rate).to(device)
    
    # We use Cosine Embedding Loss, which tries to make vectors similar
    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_mrr = 0.0
    epochs_no_improve = 0
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            
            # Cosine loss needs a third arg 'target' (1 for similar, -1 for dissimilar)
            target = torch.ones(x_batch.size(0)).to(device) 
            loss = loss_fn(y_pred, y_batch, target)
            
            loss.backward()
            optimizer.step()
            
        # 5. Validation Loop
        model.eval()
        all_y_pred_val = []
        all_y_true_val = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_pred = model(x_batch)
                all_y_pred_val.append(y_pred.cpu().numpy())
                all_y_true_val.append(y_batch.numpy())
        
        y_pred_vectors = np.concatenate(all_y_pred_val)
        y_true_vectors = np.concatenate(all_y_true_val)
        
        # Calculate our *real* metric: MRR
        val_mrr = calculate_mrr(y_pred_vectors, y_true_vectors, Y_target_all)
        
        # 6. Report to Optuna
        trial.report(val_mrr, epoch)
        
        # Handle pruning (stop bad trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        # Early Stopping
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= 5: # 5 epochs of no improvement
            print(f"Trial {trial.number}: Early stopping at epoch {epoch}")
            break
            
    # 7. Return the best validation score for this trial
    return best_val_mrr

# --- Main Execution ---

def main():
    args = parse_args()
    
    # 1. Load all data based on arguments
    try:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_target_all = load_data(
            args.features_dir, args.use_text, args.use_skill_text, args.use_structured
        )
    except FileNotFoundError:
        print(f"Error: Could not find feature files in {args.features_dir}")
        print("Please run generate_features.py first.")
        return
    except ValueError as e:
        print(e)
        return

    # 2. Create an Optuna study
    # We want to maximize the validation MRR
    study = optuna.create_study(direction="maximize")
    
    print(f"Starting Optuna study with {args.n_trials} trials...")
    start_time = time.time()
    
    # 3. Run the optimization
    # We pass our data into the objective function
    study.optimize(
        lambda trial: objective(trial, X_train, Y_train, X_val, Y_val, Y_target_all, args),
        n_trials=args.n_trials
    )
    
    end_time = time.time()
    print(f"\nOptuna study finished in {(end_time - start_time) / 60:.2f} minutes.")
    
    # 4. Print results
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Validation MRR: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # 5. Train final model and evaluate on Test Set
    print("\nTraining final model with best hyperparameters on (train + val) data...")
    
    # Combine train and val sets
    X_train_final = np.concatenate([X_train, X_val])
    Y_train_final = np.concatenate([Y_train, Y_val])
    
    train_dataset = TensorDataset(torch.tensor(X_train_final).float(), torch.tensor(Y_train_final).float())
    # Use the best batch size from the study
    final_batch_size = study.best_params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True)
    
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=final_batch_size * 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(
        input_dim=X_train_final.shape[1],
        output_dim=Y_train_final.shape[1],
        n_layers=study.best_params["n_layers"],
        hidden_dim=study.best_params["hidden_dim"],
        dropout_rate=study.best_params["dropout_rate"]
    ).to(device)
    
    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=study.best_params["lr"])

    # Train for the full number of epochs (no early stopping on test set)
    for epoch in range(args.epochs):
        model.train()
        for x_batch, y_batch in tqdm(train_loader, desc=f"Final Train Epoch {epoch+1}/{args.epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            target = torch.ones(x_batch.size(0)).to(device)
            loss = loss_fn(y_pred, y_batch, target)
            loss.backward()
            optimizer.step()
            
    # 6. Final Evaluation on Test Set
    model.eval()
    all_y_pred_test = []
    all_y_true_test = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Final Test Evaluation"):
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            all_y_pred_test.append(y_pred.cpu().numpy())
            all_y_true_test.append(y_batch.numpy())
            
    y_pred_vectors = np.concatenate(all_y_pred_test)
    y_true_vectors = np.concatenate(all_y_true_test)
    
    # We use Y_target_all, which contains all possible targets
    test_mrr = calculate_mrr(y_pred_vectors, y_true_vectors, Y_target_all)
    
    print("\n--- FINAL TEST SET RESULTS ---")
    print(f"Test MRR: {test_mrr:.4f}")
    
    # (Here you would add R@5, R@10 calculations to the `calculate_mrr` function)

if __name__ == "__main__":
    main()