import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, layers=[16, 8], dropout=False):
        """
        Initializes an instance of the NCF class.

        Parameters:
            num_users (int): The number of users in the dataset.
            num_items (int): The number of items in the dataset.
            embed_dim (int): The dimension of the embedding vectors.
            layers (list, optional): A list specifying the sizes of the MLP layers. Defaults to [16, 8].
            dropout (bool, optional): A flag indicating whether to use dropout in the MLP layers. Defaults to False.

        Returns:
            None
        """
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # GMF embeddings
        self.user_gmf_embedding = nn.Embedding(num_users, embed_dim)
        self.item_gmf_embedding = nn.Embedding(num_items, embed_dim)
        
        # MLP embeddings
        self.user_mlp_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_mlp_embedding = nn.Embedding(num_items, layers[0] // 2)
        
        # MLP layers
        mlp_modules = []
        for i, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            mlp_modules.append(nn.Linear(in_size, out_size))
            mlp_modules.append(nn.ReLU())
            if dropout:
                mlp_modules.append(nn.Dropout(p=0.2))
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        # Final prediction layer
        self.prediction_layer = nn.Linear(layers[-1] + embed_dim, 1)
        
    def forward(self, user_indices, item_indices):
        user_gmf_embed = self.user_gmf_embedding(user_indices)
        item_gmf_embed = self.item_gmf_embedding(item_indices)
        
        user_mlp_embed = self.user_mlp_embedding(user_indices)
        item_mlp_embed = self.item_mlp_embedding(item_indices)
        
        # GMF part
        gmf_product = torch.mul(user_gmf_embed, item_gmf_embed)
        
        # MLP part
        mlp_input = torch.cat([user_mlp_embed, item_mlp_embed], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Final layer
        final_input = torch.cat([gmf_product, mlp_output], dim=1)
        prediction = self.prediction_layer(final_input)
        
        return prediction.squeeze(-1)