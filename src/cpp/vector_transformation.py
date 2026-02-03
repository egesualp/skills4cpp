import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DataParallel
import argparse
from config_utils import load_train_config
from data_classes import Data


def save_model(model, model_path, input_size, hidden_sizes, output_size):
    """
    Save the model state dictionary along with additional parameters.

    Args:
        model (nn.Module): Trained PyTorch model.
        model_path (str): Path to save the model.
        input_size (int): Size of input vectors.
        hidden_sizes (list): List of hidden layer sizes.
        output_size (int): Size of output vectors.
    """
    torch.save(
        {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "output_size": output_size,
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )


# Define the neural network model with multiple hidden layers
class VectorTransformModel(nn.Module):
    """
    Neural network model for vector transformation with multiple hidden layers.

    Args:
        input_size (int): Size of input vectors.
        hidden_sizes (list): List of hidden layer sizes.
        output_size (int): Size of output vectors.
        dropout (bool): Whether to use dropout.
        dropout_rate (float): Dropout rate if dropout is enabled.
    """
    def __init__(
        self, input_size, hidden_sizes, output_size, dropout=False, dropout_rate=0.1
    ):
        super(VectorTransformModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size

        # Create hidden layers based on provided hidden sizes
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.dropout = dropout
        self.dropout_rate = dropout_rate
        if self.dropout:
            # Create dropout layer
            self.dropout = nn.Dropout(p=self.dropout_rate)

        # Output layer
        self.output = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            if self.dropout:
                x = self.dropout(x)
        x = self.output(x)
        return x


def train(config):
    # Load pre-trained sentence transformer model
    model = SentenceTransformer(config["model"]["embedding_model_transformation"])

    # Data
    ### Load data for neural transformation training
    print("Loading data...")
    data = Data(
        config["data"]["data_type"]
    )

    train_pairs, val_pairs, _ = data.get_data(stage="transformation_finetuning")

    # Define a function to create a DataLoader object
    def create_data_loader(pairs):
        # Example career history and ESCO occupation descriptions
        career_history_texts, esco_occupation_texts = zip(*pairs)

        print("Embedding career history and ESCO occupation texts...")

        # Embed career history texts
        career_history_embeddings = model.encode(career_history_texts)
        # Save the rows of the matrix in a list
        career_history_embeddings = career_history_embeddings.tolist()

        # Embed ESCO occupation texts
        esco_occupation_embeddings = model.encode(esco_occupation_texts)
        # Save the rows of the matrix in a list
        esco_occupation_embeddings = esco_occupation_embeddings.tolist()

        print("Setting up the neural network model...")

        # Reshape the data into numpy arrays
        L1 = np.array(career_history_embeddings)
        L2 = np.array(esco_occupation_embeddings)

        # Convert the numpy arrays to PyTorch tensors
        L1_tensor = torch.tensor(L1, dtype=torch.float32)
        L2_tensor = torch.tensor(L2, dtype=torch.float32)

        dataset = TensorDataset(L1_tensor, L2_tensor)
        # If more than 1 GPU is available, use DataParallel
        if torch.cuda.device_count() > 1:
            loader = DataLoader(
                dataset,
                batch_size=config["neural"]["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset, batch_size=config["neural"]["batch_size"], shuffle=True
            )

        return loader, L1.shape[1], L2.shape[1]

    train_loader, input_size, output_size = create_data_loader(train_pairs)
    test_loader, _, _ = create_data_loader(val_pairs)

    # Define hidden sizes
    hidden_sizes = config["neural"]["hidden_sizes"]

    # Define dropout
    dropout = config["neural"]["dropout"]
    dropout_rate = config["neural"]["dropout_rate"]

    # Initialize the model, loss function, and optimizer
    model = VectorTransformModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=dropout,
        dropout_rate=dropout_rate,
    )
    criterion = nn.CosineEmbeddingLoss()  # Using cosine similarity loss
    optimizer = optim.Adam(model.parameters(), lr=config["neural"]["learning_rate"])

    # Enable multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = DataParallel(model)

    print("Training the model...")

    # Train the model
    num_epochs = config["neural"]["epochs"]
    best_loss = float("inf")
    patience = config["neural"]["patience"]
    early_stop_counter = 0

    best_model = None
    best_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # Set the model to the GPU
        model = model.cuda()

        # Set the model to training mode
        model.train()
        total_loss = 0.0

        for inputs, targets in tqdm(train_loader):
            # Move the input and target tensors to the GPU
            inputs = inputs.cuda()
            targets = targets.cuda()

            # Normalize the input
            inputs_normalized = inputs / torch.norm(inputs, dim=1, keepdim=True)

            # Forward pass
            outputs = model(inputs_normalized)
            labels = torch.ones(
                inputs.size(0)
            ).cuda()  # CosineEmbeddingLoss expects labels of 1 for similar pairs
            loss = criterion(outputs, targets, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}"
        )

        # Evaluate the model on the test dataset
        model.eval()
        total_loss_test = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move the input and target tensors to the GPU
                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = model(inputs)
                targets_normalized = targets / torch.norm(targets, dim=1, keepdim=True)
                outputs_normalized = outputs / torch.norm(outputs, dim=1, keepdim=True)
                labels = torch.ones(inputs.size(0)).cuda()
                loss = criterion(outputs_normalized, targets_normalized, labels)
                total_loss_test += loss.item()

        avg_loss_test = total_loss_test / len(test_loader)
        print(f"Test Loss: {avg_loss_test:.4f}")

        # Check for early stopping
        if avg_loss_test < best_loss:
            best_loss = avg_loss_test
            best_model = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break

    # Load the best model
    model.load_state_dict(best_model)

    print("Finished training!")
    print("Best test loss:", best_loss)

    # Evaluate the model
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move the input and target tensors to the GPU
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            targets_normalized = targets / torch.norm(targets, dim=1, keepdim=True)
            outputs_normalized = outputs / torch.norm(outputs, dim=1, keepdim=True)
            labels = torch.ones(inputs.size(0)).cuda()
            loss = criterion(outputs_normalized, targets_normalized, labels)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

    # Save the model
    save_model(model, config["output"]["path_neural_transformation_model"], input_size, hidden_sizes, output_size)

    print("Saved vector transformation model...")

def main(config):
    train(config)

if __name__ == "__main__":
    # Argparse python src/vector_transformation.py --vector_transformation_config decorte_esco.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_transformation_config", type=str)
    args = parser.parse_args()
    config = load_train_config(args.vector_transformation_config)
    main(config)
