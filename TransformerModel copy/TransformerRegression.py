import torch
from transformers import BertModel, BertConfig
import torch.nn as nn
import torch.optim as optim

class TransformerRegression(nn.Module):
    def __init__(self, input_dim, output_dim, bert_config):
        super(TransformerRegression, self).__init__()

        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs[1])  # Use the [CLS] token representation
        output = self.linear(pooled_output)
        return output

# Example usage:

# Generate some random data for demonstration
input_ids = torch.randint(0, 1000, (10, 5))  # Batch size of 10, sequence length of 5
attention_mask = torch.ones_like(input_ids)  # All tokens are attended to

# Model parameters
input_dim = 768  # Hidden size of BERT models
output_dim = 1  # Dimension of output for regression

# Load BERT configuration
bert_config = BertConfig.from_pretrained('bert-base-uncased')

# Initialize model
model = TransformerRegression(input_dim, output_dim, bert_config)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(input_ids, attention_mask)
    target_data = torch.randn(10, 1)  # Random target data for demonstration
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')