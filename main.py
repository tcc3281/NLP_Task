import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# import pandas as pd

class RNN_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, output_dim, dropout):
        super(RNN_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        # xu ly chuoi co do dai khac nhau
        packed_embedded=nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        # Get the last hidden state (from the last layer)
        # Shape: [batch_size, hidden_dim]
        hidden = hidden[-1, :, :]
        
        # Apply dropout and pass through the linear layer
        hidden = self.dropout(hidden)
        
        # Pass through final linear layer
        output = self.fc(hidden)
        
        # Apply sigmoid for binary classification
        return self.sigmoid(output)
    
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]\
        
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encoding, labels, _ = self.data.iloc[idx]['tokenized_data']
        
        # Chuyển đổi các giá trị tensor thành dict
        item = {key: val.squeeze() for key, val in encoding.items() if key != 'offset_mapping'}
        item['labels'] = torch.tensor(labels)
        
        return item