# Pretraining the TCR model...
# Epoch 0, Loss 1.6008322450769954
# Epoch 1, Loss 1.4935913423457778
# Epoch 2, Loss 1.4672028950179916
# Pretraining the antigen model...
# Epoch 0, Loss 0.07184380341895592
# Epoch 1, Loss 0.036809649607566386
# Epoch 2, Loss 0.03450142668107109
#Fold 1 accuracy: 100%
#Fold 2 accuracy: 100%
#Fold 3 accuracy: 100%
#Average accuracy across all folds: 100.00%
#Average accuracy without pretraining: 100.00%

#Everytime I tried to run the program in full with 10 epochs with the full data.csv, it would take 8 hours or more or my laptop would crash, so I shortened the data for about 32 thousand lines and ran the code twice with 3 epochs each for pre-training and training, the acuuarcy came out to be 100 which was expected
#I know my dataset size after decreasing  it is was not enough to test this so I expcetd the answeer, but I my laptops computation power for Transformers was not enough 
#I did try with the original data.csv dataset, it ran till the first training  fold and the accuracy was 75>3% which took 6 hours but after that it crashed.
#I tried several times on the original data to pretrain and train it but it would take so much of my CPU and due to this my laptop performance started to slow.
#This is why I decided to shorten the original data data.

import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np

# Load the data
data = pd.read_csv('yoyo.csv')

# Define the set of amino acids and the padding character
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
token2int = {token: i for i, token in enumerate(amino_acids)}

def preprocess_sequences(sequences):
    preprocessed_sequences = []
    for seq in sequences:
        preprocessed_seq = ''.join(token if token in token2int else 'X' for token in seq)
        preprocessed_sequences.append(preprocessed_seq)
    return preprocessed_sequences

def pad_sequences(sequences, max_length, pad_token='X'):
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq.ljust(max_length, pad_token)
        padded_sequences.append(padded_seq)
    return padded_sequences

def make_tfm_model(T):
    # Define the transformer model
    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

    # Define the embedding layer
    embedding = nn.Embedding(21, 256)  # 20 amino acids + 1 padding character

    # Combine the embedding layer and the transformer encoder into a single nn.Module
    model = nn.Sequential(
        embedding,
        transformer_encoder
    )

    return model

class PredictModel(nn.Module):
    def __init__(self, M_antigen, M_tcr):
        super(PredictModel, self).__init__()
        self.M_antigen = M_antigen
        self.M_tcr = M_tcr
        self.classifier = nn.Linear(256*2, 2)

    def forward(self, antigen, tcr):
        antigen = self.M_antigen(antigen)
        tcr = self.M_tcr(tcr)
        x = torch.cat((antigen[:, -1], tcr[:, -1]), dim=-1)  # Concatenate the last output vectors
        x = x.squeeze(1)  # Remove the extra dimension
        x = self.classifier(x)
        return x

def make_predict_model(M_antigen, M_tcr):
    return PredictModel(M_antigen, M_tcr)

def pretrain_tfm_model(M, sequences, n_epochs):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(M.parameters())

    # Create a DataLoader for batching the data
    dataloader = DataLoader(sequences, batch_size=32, shuffle=True)

    # For each epoch...
    for epoch in range(n_epochs):
        total_loss = 0.0
        for i, (input_seq, target_seq) in enumerate(dataloader):
            # Forward pass
            output = M(input_seq)
            output = output.view(-1, output.size(-1))  # Reshape for loss computation
            target_seq = target_seq.view(-1)  # Reshape for loss computation

            # Compute the loss
            loss = criterion(output, target_seq)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the average loss for this epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Loss {avg_loss}')

def train_model(M, L_antigen, L_tcr, Interaction, n_epochs):
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Combine the data into a single dataset
    dataset = list(zip(L_antigen, L_tcr, Interaction))

    # Initialize the KFold cross-validator
    kf = KFold(n_splits=3, shuffle=True)

    accuracies = []
    best_models = []

    # For each fold...
    for fold, (train_index, test_index) in enumerate(kf.split(dataset), 1):
        print(f"Fold {fold}")
        # Split the data into training and test sets
        train_data = [dataset[i] for i in train_index]
        test_data = [dataset[i] for i in test_index]

        # Create DataLoaders for the training and test sets
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Initialize the model
        model = make_predict_model(make_tfm_model('antigen'), make_tfm_model('TCR'))

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters())

        # For each epoch...
        for epoch in range(n_epochs):
            # For each batch in the training data...
            for i, (antigen, tcr, Interaction) in enumerate(train_dataloader):
                # Forward pass
                output = model(antigen, tcr)
                loss = criterion(output, Interaction.float())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate the model on the test data
        correct = 0
        total = 0
        with torch.no_grad():
            for antigen, tcr, Interaction in test_dataloader:
                output = model(antigen, tcr)
                predicted = torch.round(torch.sigmoid(output))
        
                # Convert one-hot encoded vectors to class labels
                predicted_labels = torch.argmax(predicted, dim=1)
                true_labels = torch.argmax(Interaction, dim=1)
        
                # Update correct predictions count
                correct += (predicted_labels == true_labels).sum().item()
        
                # Update total count
                total += Interaction.size(0)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Fold {fold} accuracy: {accuracy:.2f}%")
        best_models.append(model.state_dict())  # Save the best model for this fold

    # Calculate the average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    print(f'Average accuracy across all folds: {avg_accuracy:.2f}%')

    # Save the best model from the fold with the highest accuracy
    best_model_idx = np.argmax(accuracies)
    best_model_state_dict = best_models[best_model_idx]
    torch.save(best_model_state_dict, 'model.pth')

    return avg_accuracy


  

def predict(M, L_antigen, L_tcr):
    predictions = []
    with torch.no_grad():
        for antigen, tcr in zip(L_antigen, L_tcr):
            antigen = torch.tensor([token2int[token] for token in antigen], dtype=torch.long)
            tcr = torch.tensor([token2int[token] for token in tcr], dtype=torch.long)
            output = M(antigen.unsqueeze(0), tcr.unsqueeze(0))
            prediction = torch.round(torch.sigmoid(output)).item()
            predictions.append(int(prediction))
    return predictions

def load_trained_model(max_len):
    model =  make_tfm_model()
    model_url = "https://drive.google.com/file/d/13sHVwfL1PFqbkpgOibvIOrjw9eOzfjRo/view?usp=sharing"  
   
    
    # Download the model file
    response = requests.get(model_url)
    with open("model_state_dict.pth", "wb") as f:
        f.write(response.content)
    
    # Load the downloaded model file
    model.load_state_dict(torch.load("model_state_dict.pth"))
    
    return model

if __name__ == "__main__":
 
    # Load the dataset
    L_antigen, L_tcr, Interaction = data['antigen'].tolist(), data['TCR'].tolist(), data['interaction'].tolist()

    # Preprocess the sequences
    L_antigen = preprocess_sequences(L_antigen)
    L_tcr = preprocess_sequences(L_tcr)

    # Pad the sequences
    all_sequences = L_antigen + L_tcr
    max_length = max(len(seq) for seq in all_sequences)
    L_antigen = pad_sequences(L_antigen, max_length)
    L_tcr = pad_sequences(L_tcr, max_length)

    # Convert the sequences to PyTorch tensors
    L_antigen = [torch.tensor([token2int[token] for token in seq], dtype=torch.long) for seq in L_antigen]
    L_tcr = [torch.tensor([token2int[token] for token in seq], dtype=torch.long) for seq in L_tcr]
    # Convert Interaction to PyTorch tensor before one-hot encoding
    Interaction = torch.tensor(Interaction)

    # Apply one-hot encoding
    Interaction = torch.nn.functional.one_hot(Interaction.to(torch.int64), num_classes=2).to(torch.float32)
    # Prepare the pretraining data
    pretraining_data_antigen = list(zip(L_antigen[:-1], L_antigen[1:]))
    pretraining_data_tcr = list(zip(L_tcr[:-1], L_tcr[1:]))

    # Define the models
    m_tcr = make_tfm_model('TCR')
    m_ant = make_tfm_model('antigen')

    # Pretrain the models
    print("Pretraining the TCR model...")
    pretrain_tfm_model(m_tcr, pretraining_data_tcr, 10)
    print("Pretraining the antigen model...")
    pretrain_tfm_model(m_ant, pretraining_data_antigen, 10)

    # Create the full prediction model
    model = make_predict_model(m_ant, m_tcr)

    # Train the full prediction model
    print("Training the full prediction model...")
    avg_accuracy = train_model(model, L_antigen, L_tcr, Interaction, 10)


    
