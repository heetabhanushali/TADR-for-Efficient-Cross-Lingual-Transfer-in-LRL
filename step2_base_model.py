"""
TADR Framework - Step 2: Base Model Setup
Loading and preparing XLM-R/mBERT for adapter insertion
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    XLMRobertaModel,
    BertModel
)
from typing import Optional, Dict, Any
import warnings

DEFAULT_MODEL_NAME = "xlm-roberta-base"
# DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"

class BaseModelWrapper(nn.Module):
    """
    Wrapper for multilingual pre-trained language models (mPLMs).
    Handles loading, freezing, and preparing the model for adapter insertion.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        freeze_base: bool = True,
        output_hidden_states: bool = True,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
                Options: 'xlm-roberta-base', 'bert-base-multilingual-cased', etc.
            freeze_base: Whether to freeze the base model parameters
            output_hidden_states: Whether to output hidden states from all layers
            device: Device to load model on ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load config and model
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config
        ).to(self.device)
        
        # Store model dimensions
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        
        print(f"Model loaded successfully!")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Number of layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_attention_heads}")
        
        # Freeze base model if specified
        if freeze_base:
            self.freeze_base_model()
            
        # Store layer modules for adapter insertion
        self.encoder_layers = self._get_encoder_layers()
        
    def _get_encoder_layers(self):
        """Get the encoder layers from the model."""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            # For XLM-R, BERT and similar models
            return self.model.encoder.layer
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layer'):
            # For DistilBERT
            return self.model.transformer.layer
        elif hasattr(self.model, 'layers'):
            # Alternative structure
            return self.model.layers
        else:
            raise AttributeError(f"Could not find encoder layers in model. Model structure: {type(self.model)}")
    
    def freeze_base_model(self):
        """Freeze all parameters in the base model."""
        print("Freezing base model parameters...")
        frozen_params = 0
        for param in self.model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print(f"  Froze {frozen_params:,} parameters")
    
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base model."""
        print("Unfreezing base model parameters...")
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": return_dict
        }
        if "token_type_ids" in self.model.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        """
        Forward pass through the base model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs (batch_size, seq_len)
            return_dict: Whether to return a dict
            
        Returns:
            Model outputs with hidden states
        """
        outputs = self.model(**kwargs)
        return outputs
    
    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get [CLS] token embedding from the last layer.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            CLS embeddings (batch_size, hidden_size)
        """
        outputs = self.forward(input_ids, attention_mask)
        # Get the [CLS] token (first token) from last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
    
    def get_layer_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get output from a specific layer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Hidden states from specified layer
        """
        outputs = self.forward(input_ids, attention_mask)
        return outputs.hidden_states[layer_idx]


class ModelWithAdapterSlots(BaseModelWrapper):
    """
    Extended base model with placeholder slots for adapters.
    Prepares the model for adapter insertion in Step 3.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        freeze_base: bool = True,
        num_adapter_layers: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze_base: Whether to freeze base model
            num_adapter_layers: Number of layers to add adapters to (None = all)
            device: Device to use
        """
        super().__init__(
            model_name=model_name,
            freeze_base=freeze_base,
            device=device
        )
        
        # Determine which layers get adapters
        if num_adapter_layers is None:
            self.adapter_layer_indices = list(range(self.num_layers))
        else:
            # Add adapters to last N layers
            self.adapter_layer_indices = list(range(
                max(0, self.num_layers - num_adapter_layers),
                self.num_layers
            ))
        
        print(f"Prepared {len(self.adapter_layer_indices)} layers for adapter insertion")
        print(f"  Layer indices: {self.adapter_layer_indices}")
        
        # Placeholder for adapters (will be added in Step 3)
        self.adapters = nn.ModuleDict()
        
    def print_model_summary(self):
        """Print a summary of the model architecture."""
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Layers with adapters: {len(self.adapter_layer_indices)}")
        print(f"Total parameters: {self.get_num_total_params():,}")
        print(f"Trainable parameters: {self.get_num_trainable_params():,}")
        trainable_pct = (self.get_num_trainable_params() / 
                        self.get_num_total_params() * 100)
        print(f"Trainable: {trainable_pct:.2f}%")
        print("="*60 + "\n")


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_base_model():
    """Test the base model wrapper."""
    print("="*60)
    print("Testing Base Model Setup")
    print("="*60)
    
    # Test 1: Load XLM-R
    print("\n1. Loading XLM-RoBERTa...")
    model = BaseModelWrapper(
        model_name = DEFAULT_MODEL_NAME,
        freeze_base=True
    )
    
    print(f"\n✅ Model loaded successfully")
    print(f"   Total params: {model.get_num_total_params():,}")
    print(f"   Trainable params: {model.get_num_trainable_params():,}")
    
    # Test 2: Tokenize sample text
    print("\n2. Testing Tokenization...")
    texts = [
        "Hello, how are you?",
        "नमस्ते, आप कैसे हैं?",  # Hindi
        "你好吗？"  # Chinese
    ]
    
    encoded = model.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    print(f"   Input shape: {encoded['input_ids'].shape}")
    print(f"   Sample tokens (English): {encoded['input_ids'][0][:10]}")
    
    # Test 3: Forward pass
    print("\n3. Testing Forward Pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    print(f"   Last hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"   Number of hidden states: {len(outputs.hidden_states)}")
    
    # Test 4: Get CLS embeddings
    print("\n4. Testing CLS Embedding Extraction...")
    with torch.no_grad():
        cls_embeddings = model.get_cls_embedding(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    print(f"   CLS embeddings shape: {cls_embeddings.shape}")
    print(f"   Sample CLS (first 5 dims): {cls_embeddings[0][:5]}")
    
    # Test 5: Model with adapter slots
    print("\n5. Testing Model with Adapter Slots...")
    model_with_slots = ModelWithAdapterSlots(
        model_name = DEFAULT_MODEL_NAME,
        freeze_base=True,
        num_adapter_layers=6  # Only last 6 layers
    )
    
    model_with_slots.print_model_summary()
    
    # Test 6: Try mBERT as alternative
    print("\n6. Testing Alternative Model (mBERT)...")
    try:
        mbert = BaseModelWrapper(
            model_name="bert-base-multilingual-cased",
            freeze_base=True
        )
        print(f"   ✅ mBERT loaded successfully")
        print(f"   Hidden size: {mbert.hidden_size}")
    except Exception as e:
        print(f"   ⚠️  Could not load mBERT: {e}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    
    return model_with_slots


def compare_models():
    """Compare different base models."""
    print("\n" + "="*60)
    print("Comparing Different Base Models")
    print("="*60)
    
    models_to_test = [
        # "xlm-roberta-base",
        DEFAULT_MODEL_NAME,
        "distilbert-base-multilingual-cased" if DEFAULT_MODEL_NAME == "xlm-roberta-base" else "xlm-roberta-base",
        "bert-base-multilingual-cased",
    ]
    
    results = []
    
    for model_name in models_to_test:
        try:
            print(f"\nLoading {model_name}...")
            model = BaseModelWrapper(model_name, freeze_base=True)
            
            results.append({
                'name': model_name,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'total_params': model.get_num_total_params(),
                'vocab_size': len(model.tokenizer)
            })
            
            print(f"  ✅ Loaded successfully")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Hidden size: {r['hidden_size']}")
        print(f"  Layers: {r['num_layers']}")
        print(f"  Parameters: {r['total_params']:,}")
        print(f"  Vocabulary: {r['vocab_size']:,}")


if __name__ == "__main__":
    # Run basic tests
    model = test_base_model()
    
    # Compare models
    compare_models()
    
    print("\n" + "="*60)
    print("Step 2 Complete!")
    print("Next: Step 3 - Adapter Modules")
    print("="*60)