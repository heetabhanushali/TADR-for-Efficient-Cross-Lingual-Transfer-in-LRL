"""
TADR Framework - Step 1: Typological Feature Module
Complete implementation for loading and embedding typological features
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class TypologicalFeatureLoader:
    """
    Loads and manages typological features for languages.
    Features can come from WALS or be computationally derived.
    """
    
    def __init__(self, feature_file: str, feature_names: Optional[List[str]] = None , missing_value_strategy: str = 'zero'):
        """
        Args:
            feature_file: Path to CSV/JSON with typological features
            feature_names: Specific features to use (or None for all)
        """
        self.feature_file = feature_file
        self.feature_data = self._load_features(feature_file)
        self.feature_names = feature_names or self._get_all_features()
        self.feature_dim = len(self.feature_names)
        self.missing_value_strategy = missing_value_strategy
        
        print(f"Loaded {len(self.feature_data)} languages")
        print(f"Feature dimension: {self.feature_dim}")
        
    def _load_features(self, file_path: str) -> pd.DataFrame:
        """Load features from file."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
    
    def _get_all_features(self) -> List[str]:
        """Get all feature column names."""
        # Exclude metadata columns
        exclude = ['language', 'iso_code', 'lang_id', 'name', 'latitude', 'longitude']
        return [col for col in self.feature_data.columns if col not in exclude]
    
    def get_feature_vector(self, lang_id: str) -> np.ndarray:
        """
        Get feature vector for a language.
        
        Args:
            lang_id: Language identifier (ISO code)
            
        Returns:
            Feature vector with shape (feature_dim,)
        """
        lang_data = self.feature_data[self.feature_data['iso_code'] == lang_id]
        
        if len(lang_data) == 0:
            print(f"Warning: Language '{lang_id}' not found. Returning zero vector.")
            return np.zeros(self.feature_dim)
        
        features = lang_data[self.feature_names].values[0]
        
        # Handle missing values (NaN) with zeros
        if self.missing_value_strategy == 'zero':
            features = np.nan_to_num(features, nan=0.0)
        elif self.missing_value_strategy == 'mean':
            for i , feat_name in enumerate(self.feature_names):
                if np.isnan(features[i]):
                    mean_val = self.feature_data[feat_name].mean()
                    features[i] = mean_val if not np.isnan(mean_val) else 0.0
        
        return features.astype(np.float32)
    
    def get_all_languages(self) -> List[str]:
        """Get list of all available language IDs."""
        return self.feature_data['iso_code'].tolist()
    
    def create_language_mapping(self) -> Dict[str, int]:
        """Create mapping from language ID to index."""
        languages = self.get_all_languages()
        return {lang: idx for idx, lang in enumerate(languages)}


class TypologyEmbedding(nn.Module):
    """
    Learnable embedding layer for typological features.
    Transforms raw features into dense representations.
    """
    
    def __init__(self, 
                 feature_dim: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 use_feature_weights: bool = True):
        """
        Args:
            feature_dim: Dimension of raw feature vector
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        if use_feature_weights:
            self.feature_weights = nn.Parameter(torch.ones(feature_dim))
        else:
            self.register_buffer('feature_weights', torch.ones(feature_dim))
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # MLP: feature_dim -> hidden_dim -> embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform features to embeddings.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        weighted_features = features * self.feature_weights
        return self.mlp(weighted_features)


class TypologyFeatureModule(nn.Module):
    """
    Complete module combining feature loading and embedding.
    This is the main class to use.
    """
    
    def __init__(self,
                 feature_loader: TypologicalFeatureLoader,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            feature_loader: TypologicalFeatureLoader instance
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_loader = feature_loader
        self.embedding_dim = embedding_dim
        
        # Create embedding network
        self.embedding_net = TypologyEmbedding(
            feature_dim=feature_loader.feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Cache for embeddings (optimization for inference)
        self.embedding_cache = {}
        
    def get_embedding(self, lang_id: str, use_cache: bool = True) -> torch.Tensor:
        """
        Get typology embedding for a language.
        
        Args:
            lang_id: Language identifier
            use_cache: Whether to use cached embeddings
            
        Returns:
            embedding: Tensor of shape (embedding_dim,)
        """
        if use_cache and lang_id in self.embedding_cache:
            return self.embedding_cache[lang_id]
        
        # Get raw features
        features = self.feature_loader.get_feature_vector(lang_id)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Compute embedding
        with torch.no_grad():
            embedding = self.embedding_net(features_tensor).squeeze(0)
        
        if use_cache:
            self.embedding_cache[lang_id] = embedding
            
        return embedding
    
    def forward(self, lang_ids: List[str]) -> torch.Tensor:
        """
        Batch processing of language embeddings.
        
        Args:
            lang_ids: List of language identifiers
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        embeddings = []
        for lang_id in lang_ids:
            emb = self.get_embedding(lang_id, use_cache=False)
            embeddings.append(emb)
        
        return torch.stack(embeddings)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache = {}

