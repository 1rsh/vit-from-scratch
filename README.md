# Vision Transformer Implementation for Devanagari Character Recognition ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

![vit-header](https://github.com/1rsh/vit-from-scratch/assets/93649948/93a1b220-9b86-438f-9d23-b57473eb3ecc)

This repository contains an implementation of ViT (Vision Transformer Architecture) that is introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) using PyTorch.  
<br>
The model is trained on [Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) available on the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) on an M1 Pro achieving a validation accuracy of 96.70%.
<br><br>
****
## Repository Walkthrough

### 1. PatchEmbedding
```python
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        
        # Dividing into patches
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = embed_dim,
                kernel_size = patch_size,
                stride = patch_size
            ),
            nn.Flatten(2))
        
        self.cls_token = nn.Parameter(torch.randn(size = (1, in_channels, embed_dim)), requires_grad = True)
        self.position_embeddings = nn.Parameter(torch.randn(size = (1, num_patches + 1, embed_dim), requires_grad = True))
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim = 1) # adding cls_token to left
        x = self.position_embeddings + x # adding position embeddings to patches
        x = self.dropout(x)
        return x
```
### 2. VisionTransformer
```python
class VisionTransformer(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, dropout = dropout, activation = activation, batch_first = True, norm_first = True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers = num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = embed_dim),
            nn.Linear(in_features = embed_dim, out_features = num_classes)
        )
        
    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :]) # taking only cls_token
        return x
```
### 3. Hyperparameters
```python
batch_size = 512
num_epochs = 40

learning_rate = 1e-4
num_classes = 46
patch_size = 4
img_size = 32
in_channels = 1
num_heads = 8
dropout = 0.001
hidden_dim = 1024
adam_weight_decay = 0
adam_betas = (0.9, 0.999)
activation = "gelu"
num_encoders = 4
embed_dim = (patch_size ** 2) * in_channels # 16
num_patches = (img_size // patch_size) ** 2 # 64
```
### 4. Training
```python
for epoch in tqdm(range(num_epochs), position = 0, leave = True):
    model.train()
    
    train_labels = []
    train_preds = []
    
    train_running_loss = 0
    
    for idx, img_label in enumerate(tqdm(train_dataloader, position = 0, leave = True)):
        img = img_label[0].float().to(device)
        label = img_label[1].type(torch.uint8).to(device)
        
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim = 1)
        
        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())
        
        loss = criterion(y_pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        
    train_loss = train_running_loss / (idx + 1)
    
    
    if((epoch + 1) % 5 == 0):
        model.eval()

        val_labels = []
        val_preds = []
        val_running_loss = 0

        with torch.no_grad():
            for idx, img_label in enumerate(tqdm(test_dataloader, position = 0, leave = True)):
                img = img_label[0].float().to(device)
                label = img_label[1].type(torch.uint8).to(device)

                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim = 1)

                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())

                loss = criterion(y_pred, label)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)
```
![training-history](https://github.com/1rsh/vit-from-scratch/assets/93649948/aa15ebe9-420b-4d70-ade9-98eb10df8194)


## Footnote
If you wish to use the following PyTorch implementation of Vision Transformer for your own project, just download the notebook and update train_dir and test_dir acoording to your file hierarchy. Also make sure to adjust variables such as img_size. <br>
Feel free to contact me at <a href = "mailto:irsh.iitkgp@gmail.com">irsh.iitkgp@gmail.com</a>.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
