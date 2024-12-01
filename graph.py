import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader

class graph:
    def __init__(self, vocab_size: int, embed_size: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, context_length: int = 64, device: str = None):
        """Initialize the GPT model with configuration parameters."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {
            'vocab_size': vocab_size,
            'embed_size': embed_size,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'context_length': context_length
        }
        
        # Initialize the model
        self.model = self._build_model().to(self.device)
        self.optimizer = None
        
    def _build_model(self):
        """Construct the GPT model architecture."""
        return nn.Sequential(
            nn.Embedding(self.config['vocab_size'], self.config['embed_size']),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.config['embed_size'],
                    nhead=self.config['num_heads'],
                    dim_feedforward=4 * self.config['embed_size'],
                    batch_first=True
                ),
                num_layers=self.config['num_layers']
            ),
            nn.Linear(self.config['embed_size'], self.config['vocab_size'])
        )
    
    class _CustomDataset(Dataset):
        """Custom dataset class for handling the specified format."""
        def __init__(self, data: List[Dict], device):
            self.data = data
            self.device = device

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return (torch.tensor(item['input'], device=self.device),
                    torch.tensor(item['output'], device=self.device))

    def train(self, dataset: List[Dict], batch_size: int = 32, 
              epochs: int = 10, learning_rate: float = 0.001):
        """Train the model on the provided dataset."""
        # Initialize optimizer if not already done
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Create DataLoader
        train_dataset = self._CustomDataset(dataset, self.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, self.config['vocab_size'])
                targets = targets.view(-1)
                
                # Calculate loss
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                          f'Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

    def predict(self, input_sequence: List[int], max_length: int = None):
        """Generate predictions from an input sequence."""
        if max_length is None:
            max_length = self.config['context_length']

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_sequence, device=self.device).unsqueeze(0)
            
            for _ in range(max_length - len(input_sequence)):
                # Generate prediction
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                # Append prediction to input
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)
                
                # Optional: Break if end token is generated
                if next_token == 0:  # Assuming 0 is the end token
                    break
                
        return input_tensor.squeeze().cpu().tolist()

    def export(self, filepath: str):
        """Export the model weights and config in a JavaScript-friendly format."""
        self.model.eval()
        export_dict = {
            'config': {
                'vocab_size': self.config['vocab_size'],
                'embed_size': self.config['embed_size'],
                'num_heads': self.config['num_heads'],
                'num_layers': self.config['num_layers'],
                'context_length': self.config['context_length']
            },
            'weights': {
                'embedding.weight': self.model[0].weight.detach().cpu().numpy().tolist(),
                'transformer.layers': [],
                'output.weight': self.model[2].weight.detach().cpu().numpy().tolist(),
                'output.bias': self.model[2].bias.detach().cpu().numpy().tolist()
            }
        }
        
        # Export transformer layers
        transformer = self.model[1]
        for layer in transformer.layers:
            layer_weights = {
                'self_attn.in_proj_weight': layer.self_attn.in_proj_weight.detach().cpu().numpy().tolist(),
                'self_attn.in_proj_bias': layer.self_attn.in_proj_bias.detach().cpu().numpy().tolist(),
                'self_attn.out_proj.weight': layer.self_attn.out_proj.weight.detach().cpu().numpy().tolist(),
                'self_attn.out_proj.bias': layer.self_attn.out_proj.bias.detach().cpu().numpy().tolist(),
                'linear1.weight': layer.linear1.weight.detach().cpu().numpy().tolist(),
                'linear1.bias': layer.linear1.bias.detach().cpu().numpy().tolist(),
                'linear2.weight': layer.linear2.weight.detach().cpu().numpy().tolist(),
                'linear2.bias': layer.linear2.bias.detach().cpu().numpy().tolist(),
            }
            export_dict['weights']['transformer.layers'].append(layer_weights)
        
        with open(filepath, 'w') as f:
            json.dump(export_dict, f)
    

    def save(self, filepath: str):
        """Save the model and configuration to a file."""
        save_dict = {
            'config': self.config,
            'model_state': self.model.state_dict()
        }
        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        """Load the model and configuration from a file."""
        save_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        self.config = save_dict['config']
        self.model = self._build_model()
        self.model.load_state_dict(save_dict['model_state'])
        self.model.to(self.device)
