# utils/model_utils.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Modèle TRABSA en PyTorch
class TRABSA_PyTorch(nn.Module):
    def __init__(self, transformer_model="roberta-base", lstm_units=128, dropout_rate=0.2):
        super(TRABSA_PyTorch, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_model)
        hidden_size = self.transformer.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_units,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(lstm_units * 2, 1)  # *2 car bidirectionnel
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # Sortie du transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        
        # Passage par le BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)  # [batch_size, seq_length, lstm_units*2]
        
        # Prendre la sortie finale du BiLSTM (dernière position non masquée)
        # On peut aussi utiliser le premier token [CLS]
        lstm_output = lstm_output[:, 0, :]  # [batch_size, lstm_units*2]
        
        # Dropout pour régularisation
        lstm_output = self.dropout(lstm_output)
        
        # Classification
        logits = self.classifier(lstm_output)
        return self.sigmoid(logits)

def load_model_and_tokenizer(model_path, tokenizer_name="roberta-base", device=None):
    """Charge le modèle TRABSA et le tokenizer"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialiser le modèle
    model = TRABSA_PyTorch(transformer_model=tokenizer_name)
    
    # Charger les poids du modèle
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer, max_length=128, device=None):
    """Prédit le sentiment d'un texte avec le modèle TRABSA"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenization
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Préparer les entrées
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        proba = output.item()
    
    sentiment = "positif" if proba > 0.5 else "négatif"
    
    return sentiment, proba