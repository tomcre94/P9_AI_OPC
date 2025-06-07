import torch
import torch.nn as nn
from transformers import AutoModel

class TRABSA_PyTorch(nn.Module):
    def __init__(self, transformer_model, lstm_hidden_size=128, num_lstm_layers=1):
        super(TRABSA_PyTorch, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.dropout = nn.Dropout(0.1)

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Classifier layer: input size will be 2 * lstm_hidden_size (due to bidirectional)
        self.classifier = nn.Linear(lstm_hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid() # Add sigmoid activation

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        sequence_output = self.dropout(outputs.last_hidden_state)

        # Pass through BiLSTM
        # LSTM expects (batch_size, seq_len, input_size)
        lstm_output, _ = self.bilstm(sequence_output)

        # Take the output corresponding to the [CLS] token (first token) after BiLSTM
        pooled_output = lstm_output[:, 0, :]

        logits = self.classifier(pooled_output)
        return self.sigmoid(logits) # Apply sigmoid to the output
