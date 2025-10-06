Here are the **best methods to extract graph embeddings** from a trained PyTorch Geometric classification model:

## Method 1: Hook into the Model Before Final Classifier (Recommended)

```python
import torch
from torch_geometric.data import Data, DataLoader

class GraphEmbeddingExtractor:
    def __init__(self, model, layer_name='final_conv'):
        """
        Extract embeddings from a specific layer
        
        Args:
            model: trained PyG model
            layer_name: name of the layer to extract from (before classifier)
        """
        self.model = model
        self.model.eval()
        self.embeddings = None
        self.layer_name = layer_name
        
        # Register forward hook
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            self.embeddings = output.detach()
        
        # Find the layer to hook into
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook_fn)
                break
    
    def extract(self, data):
        """Extract embeddings for a batch of graphs"""
        with torch.no_grad():
            _ = self.model(data.x, data.edge_index, data.batch)
        return self.embeddings

# Usage
model = YourTrainedModel()
model.load_state_dict(torch.load('model.pth'))

extractor = GraphEmbeddingExtractor(model, layer_name='graph_conv3')
embeddings = extractor.extract(data)
```

## Method 2: Modify Model to Return Embeddings

```python
class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.pool = global_mean_pool
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch, return_embedding=False):
        # Node-level embeddings
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        # Graph-level embedding (pooling)
        graph_embedding = self.pool(x, batch)
        
        if return_embedding:
            return graph_embedding
        
        # Classification
        out = self.classifier(graph_embedding)
        return out

# Usage
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, data.batch, return_embedding=True)
```

## Method 3: Create Embedding Extractor from Pretrained Model

```python
class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.trained_model = trained_model
        
        # Remove the classifier layer
        if hasattr(trained_model, 'classifier'):
            self.encoder = torch.nn.Sequential(
                *list(trained_model.children())[:-1]
            )
        else:
            self.encoder = trained_model
    
    def forward(self, x, edge_index, batch):
        """Returns graph embeddings without classification"""
        with torch.no_grad():
            return self.encoder(x, edge_index, batch)

# Usage
trained_model = torch.load('full_model.pth')
extractor = EmbeddingExtractor(trained_model)
extractor.eval()

embeddings = extractor(data.x, data.edge_index, data.batch)
```

## Method 4: Extract at Different Levels (Node, Subgraph, Graph)

```python
class MultiLevelEmbeddingExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.node_embeddings = None
        self.graph_embeddings = None
    
    def extract_node_embeddings(self, data):
        """Extract node-level embeddings (before pooling)"""
        with torch.no_grad():
            x = data.x
            for layer in self.model.conv_layers:
                x = layer(x, data.edge_index).relu()
            self.node_embeddings = x
        return x
    
    def extract_graph_embeddings(self, data):
        """Extract graph-level embeddings (after pooling)"""
        node_emb = self.extract_node_embeddings(data)
        
        # Apply pooling
        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(node_emb, data.batch)
        self.graph_embeddings = graph_emb
        return graph_emb
    
    def extract_subgraph_embeddings(self, data, node_indices):
        """Extract embeddings for specific subgraphs"""
        node_emb = self.extract_node_embeddings(data)
        subgraph_emb = node_emb[node_indices]
        return subgraph_emb.mean(dim=0)  # Average pooling

# Usage
extractor = MultiLevelEmbeddingExtractor(model)
node_emb = extractor.extract_node_embeddings(data)
graph_emb = extractor.extract_graph_embeddings(data)
```

## Method 5: Batch Extraction with Different Pooling Strategies

```python
import torch
from torch_geometric.nn import (
    global_mean_pool, 
    global_max_pool, 
    global_add_pool,
    GlobalAttention
)

class GraphEmbeddingExtractorAdvanced:
    def __init__(self, model, pooling='mean'):
        self.model = model
        self.model.eval()
        
        # Choose pooling strategy
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'attention':
            hidden_dim = model.hidden_dim
            self.pool = GlobalAttention(
                torch.nn.Linear(hidden_dim, 1)
            )
    
    def extract_from_loader(self, data_loader, device='cuda'):
        """Extract embeddings for entire dataset"""
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                
                # Get node embeddings from model
                x = self.model.get_node_embeddings(data.x, data.edge_index)
                
                # Pool to graph level
                graph_emb = self.pool(x, data.batch)
                
                all_embeddings.append(graph_emb.cpu())
                all_labels.append(data.y.cpu())
        
        return torch.cat(all_embeddings), torch.cat(all_labels)

# Usage
extractor = GraphEmbeddingExtractorAdvanced(model, pooling='attention')
embeddings, labels = extractor.extract_from_loader(test_loader)
```

## Method 6: Using Graph Transformer Specific Extraction

```python
class GraphTransformerEmbedding:
    """For Graph Transformer models"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def extract_cls_token(self, data):
        """Extract [CLS] token embedding (if using virtual node)"""
        with torch.no_grad():
            output = self.model(data.x, data.edge_index, data.batch)
            # Assuming model returns dict with 'cls_token'
            if isinstance(output, dict):
                return output['cls_token']
            return output
    
    def extract_attention_pooling(self, data):
        """Extract attention-weighted graph embedding"""
        with torch.no_grad():
            # Get attention weights and node embeddings
            node_emb, attention_weights = self.model.get_attention_embeddings(
                data.x, data.edge_index, data.batch
            )
            
            # Attention-weighted pooling
            weighted_emb = node_emb * attention_weights.unsqueeze(-1)
            
            from torch_geometric.nn import global_add_pool
            graph_emb = global_add_pool(weighted_emb, data.batch)
            
        return graph_emb
```

## Method 7: Save Embeddings Efficiently

```python
import numpy as np
import h5py

class EmbeddingSaver:
    def __init__(self, model, save_path='embeddings.h5'):
        self.model = model
        self.save_path = save_path
        self.model.eval()
    
    def extract_and_save(self, data_loader, device='cuda'):
        """Extract embeddings and save to HDF5 for memory efficiency"""
        
        with h5py.File(self.save_path, 'w') as f:
            # First pass: determine embedding dimension
            sample_data = next(iter(data_loader)).to(device)
            with torch.no_grad():
                sample_emb = self.model(
                    sample_data.x, 
                    sample_data.edge_index, 
                    sample_data.batch,
                    return_embedding=True
                )
            emb_dim = sample_emb.shape[1]
            
            # Create datasets
            num_graphs = len(data_loader.dataset)
            embeddings_ds = f.create_dataset(
                'embeddings', 
                shape=(num_graphs, emb_dim),
                dtype='float32'
            )
            labels_ds = f.create_dataset(
                'labels',
                shape=(num_graphs,),
                dtype='int64'
            )
            
            # Extract and save
            idx = 0
            with torch.no_grad():
                for data in data_loader:
                    data = data.to(device)
                    emb = self.model(
                        data.x, 
                        data.edge_index, 
                        data.batch,
                        return_embedding=True
                    )
                    
                    batch_size = emb.shape[0]
                    embeddings_ds[idx:idx+batch_size] = emb.cpu().numpy()
                    labels_ds[idx:idx+batch_size] = data.y.cpu().numpy()
                    idx += batch_size
        
        print(f"Saved {num_graphs} embeddings to {self.save_path}")

# Usage
saver = EmbeddingSaver(model)
saver.extract_and_save(data_loader)

# Load later
with h5py.File('embeddings.h5', 'r') as f:
    embeddings = f['embeddings'][:]
    labels = f['labels'][:]
```

## Best Practices Summary

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Forward Hook** | Quick extraction without modifying model | No code changes | Harder to debug |
| **Return Embedding Flag** | Clean, explicit control | Easy to understand | Requires model modification |
| **Separate Encoder** | Deployment, inference | Lightweight | Need to maintain sync |
| **Multi-level** | Hierarchical analysis | Flexible | More complex |
| **Batch Extraction** | Large datasets | Memory efficient | Slower |

**Recommended approach**: Use **Method 2** (modify model to return embeddings) for the cleanest and most maintainable solution. Add a `return_embedding` flag to your forward pass during model design.