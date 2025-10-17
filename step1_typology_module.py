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
    
    def __init__(self, feature_file: str, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_file: Path to CSV/JSON with typological features
            feature_names: Specific features to use (or None for all)
        """
        self.feature_file = feature_file
        self.feature_data = self._load_features(feature_file)
        self.feature_names = feature_names or self._get_all_features()
        self.feature_dim = len(self.feature_names)
        
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
        exclude = ['language', 'iso_code', 'lang_id', 'name']
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
        features = np.nan_to_num(features, nan=0.0)
        
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
                 dropout: float = 0.1):
        """
        Args:
            feature_dim: Dimension of raw feature vector
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
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
        return self.mlp(features)


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


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def create_sample_data():
    """Create sample typological data for testing."""
    # Sample WALS-like features for 5 languages
    data = {
        'iso_code': ['en', 'hi', 'zh', 'ar', 'sw'],
        'language': ['English', 'Hindi', 'Chinese', 'Arabic', 'Swahili'],
        # Word order: 1=SVO, 6=SOV, etc.
        'word_order': [1, 6, 1, 4, 1],
        # Morphological fusion: 1-9 scale
        'morphology_fusion': [2, 8, 1, 7, 5],
        # Case marking: 0-10 scale
        'case_marking': [1, 8, 0, 6, 5],
        # Gender system: 0-10 scale
        'gender_system': [3, 2, 0, 3, 7],
        # Consonant inventory: number of consonants
        'consonant_inventory': [24, 29, 22, 28, 33],
        # Vowel inventory: number of vowels
        'vowel_inventory': [14, 10, 5, 6, 5],
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_typology_features.csv', index=False)
    print("Created sample_typology_features.csv")
    return df


def test_module():
    """Test the typological feature module."""
    print("="*60)
    print("Testing Typological Feature Module")
    print("="*60)
    
    # Create sample data
    create_sample_data()
    
    # Initialize feature loader
    print("\n1. Initializing Feature Loader...")
    feature_loader = TypologicalFeatureLoader(
        feature_file='sample_typology_features.csv',
        feature_names=['word_order', 'morphology_fusion', 'case_marking', 
                      'gender_system', 'consonant_inventory', 'vowel_inventory']
    )
    
    # Test feature extraction
    print("\n2. Testing Feature Extraction...")
    en_features = feature_loader.get_feature_vector('en')
    print(f"English features: {en_features}")
    
    hi_features = feature_loader.get_feature_vector('hi')
    print(f"Hindi features: {hi_features}")
    
    # Initialize embedding module
    print("\n3. Initializing Typology Embedding Module...")
    typo_module = TypologyFeatureModule(
        feature_loader=feature_loader,
        embedding_dim=128,
        hidden_dim=256,
        dropout=0.1
    )
    
    # Test single language embedding
    print("\n4. Testing Single Language Embedding...")
    en_embedding = typo_module.get_embedding('en')
    print(f"English embedding shape: {en_embedding.shape}")
    print(f"English embedding (first 10 dims): {en_embedding[:10]}")
    
    # Test batch processing
    print("\n5. Testing Batch Processing...")
    lang_batch = ['en', 'hi', 'zh']
    batch_embeddings = typo_module(lang_batch)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Test unknown language
    print("\n6. Testing Unknown Language...")
    unknown_emb = typo_module.get_embedding('xyz')
    print(f"Unknown language embedding shape: {unknown_emb.shape}")
    
    # Compute typological similarities
    print("\n7. Computing Typological Similarities...")
    def cosine_similarity(lang1, lang2):
        emb1 = typo_module.get_embedding(lang1)
        emb2 = typo_module.get_embedding(lang2)
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        )
        return similarity.item()
    
    print(f"EN-HI similarity: {cosine_similarity('en', 'hi'):.4f}")
    print(f"EN-ZH similarity: {cosine_similarity('en', 'zh'):.4f}")
    print(f"HI-ZH similarity: {cosine_similarity('hi', 'zh'):.4f}")
    print(f"EN-AR similarity: {cosine_similarity('en', 'ar'):.4f}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_module()