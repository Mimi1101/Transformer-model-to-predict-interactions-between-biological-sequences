import pandas as pd
import torch

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        padded_sequence = sequence + 'X' * (max_length - len(sequence))
    else:
        padded_sequence = sequence[:max_length]
    return padded_sequence

def tokenize_sequence(sequence):
    amino_acid_to_token = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20,  # Placeholder for unknown amino acids
        'B': 21,  # Represents either asparagine (N) or aspartic acid (D)
        'J': 22,  # Represents leucine (L) or isoleucine (I)
        'Z': 23,  # Represents glutamine (Q) or glutamic acid (E)
        'U': 24,  # Represents selenocysteine
        'O': 25   # Represents pyrrolysine
    }
    
    tokenized_sequence = [amino_acid_to_token.get(aa, 20) for aa in sequence]
    return tokenized_sequence

def preprocess_data(csv_file, max_length):
    data = pd.read_csv(csv_file)
    
    antigen_seqs = []
    tcr_seqs = []
    interactions = []

    for index, row in data.iterrows():
        antigen_seq = row['antigen']
        tcr_seq = row['TCR']
        interaction = row['interaction']

        antigen_seq = pad_sequence(antigen_seq, max_length)
        tcr_seq = pad_sequence(tcr_seq, max_length)

        antigen_seq = tokenize_sequence(antigen_seq)
        tcr_seq = tokenize_sequence(tcr_seq)

        antigen_seqs.append(antigen_seq)
        tcr_seqs.append(tcr_seq)
        interactions.append(interaction)

    antigen_seqs_tensor = torch.tensor(antigen_seqs)
    tcr_seqs_tensor = torch.tensor(tcr_seqs)
    interactions_tensor = torch.tensor(interactions)

    return antigen_seqs_tensor, tcr_seqs_tensor, interactions_tensor

# Example usage:
antigen_seqs, tcr_seqs, interactions = preprocess_data('data.csv', max_length=20)
print("Antigen sequences:")
print(antigen_seqs[:5])  # Print the first 5 antigen sequences
print("\nTCR sequences:")
print(tcr_seqs[:5])  # Print the first 5 TCR sequences
print("\nInteractions:")
print(interactions[:5])  # Print the first 5 interaction labels
