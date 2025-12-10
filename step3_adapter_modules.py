"""
TADR Framework - Step 3: Dynamic Adapter Modules
Implementation of bottleneck adapters with typology-based routing
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Any
import collections

# Import from previous steps
from step1_typology_module import TypologicalFeatureLoader, TypologyFeatureModule
from step2_base_model import ModelWithAdapterSlots, DEFAULT_MODEL_NAME


# ============================================================================
# OUTPUT CLASSES
# ============================================================================

TADROutput = collections.namedtuple(
    "TADROutput",
    ["last_hidden_state", "pooler_output", "hidden_states", "routing_weights"]
)


# ============================================================================
# ADAPTER COMPONENTS
# ============================================================================

class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter module with down-projection, non-linearity, and up-projection.
    
    Architecture:
        x → Down(hidden→bottleneck) → Activation → Dropout → Up(bottleneck→hidden) → LayerNorm → x + residual
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        bottleneck_size: int = 64,
        non_linearity: str = "relu",
        dropout: float = 0.1,
        init_scale: float = 1e-3,
        use_layer_norm: bool = True
    ):
        """
        Args:
            hidden_size: Dimension of the transformer hidden states
            bottleneck_size: Dimension of the bottleneck (compression dimension)
            non_linearity: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
            init_scale: Initialization scale for weights
            use_layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.reduction_factor = hidden_size / bottleneck_size
        
        # Down-projection
        self.down_project = nn.Linear(hidden_size, bottleneck_size)
        
        # Non-linearity
        if non_linearity == "relu":
            self.activation = nn.ReLU()
        elif non_linearity == "gelu":
            self.activation = nn.GELU()
        elif non_linearity == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {non_linearity}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Up-projection
        self.up_project = nn.Linear(bottleneck_size, hidden_size)
        
        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights with small values for residual stability
        self._init_weights(init_scale)
    
    def _init_weights(self, init_scale: float):
        """Initialize weights near zero for residual connection stability."""
        nn.init.normal_(self.down_project.weight, std=init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=init_scale)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Adapter output of shape (batch_size, seq_len, hidden_size)
        """
        # Store input for residual
        residual = hidden_states
        
        # Down-projection
        down = self.down_project(hidden_states)
        
        # Non-linearity
        down = self.activation(down)
        
        # Dropout
        down = self.dropout(down)
        
        # Up-projection
        up = self.up_project(down)
        
        # Layer normalization (optional)
        if self.use_layer_norm:
            up = self.layer_norm(up)
        
        # Residual connection
        output = residual + up
        
        return output
    
    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class MultiAdapterModule(nn.Module):
    """
    Module containing K adapters that are dynamically weighted based on routing.
    """
    
    def __init__(
        self,
        num_adapters: int,
        hidden_size: int = 768,
        bottleneck_size: int = 64,
        non_linearity: str = "relu",
        dropout: float = 0.1,
        adapter_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            num_adapters: Number of adapters (K)
            hidden_size: Hidden dimension
            bottleneck_size: Bottleneck dimension
            non_linearity: Activation function
            dropout: Dropout rate
            adapter_names: Optional names for adapters
        """
        super().__init__()
        
        self.num_adapters = num_adapters
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        
        # Create K adapters
        self.adapters = nn.ModuleList([
            BottleneckAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                non_linearity=non_linearity,
                dropout=dropout,
                **kwargs
            )
            for _ in range(num_adapters)
        ])
        
        # Optional adapter names for interpretability
        if adapter_names is not None:
            assert len(adapter_names) == num_adapters, \
                f"Number of names ({len(adapter_names)}) must match number of adapters ({num_adapters})"
            self.adapter_names = adapter_names
        else:
            self.adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adapters with routing weights.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            routing_weights: Routing weights (batch_size, num_adapters)
        
        Returns:
            Output tensor after weighted adapter application
        """
        # Compute output from each adapter
        adapter_outputs = []
        for adapter in self.adapters:
            output = adapter(hidden_states)
            adapter_outputs.append(output)
        
        # Stack: (num_adapters, batch_size, seq_len, hidden_size)
        adapter_outputs = torch.stack(adapter_outputs, dim=0)
        
        # Reshape routing weights for broadcasting: (num_adapters, batch_size, 1, 1)
        routing_weights = routing_weights.t().unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum: (batch_size, seq_len, hidden_size)
        weighted_output = (adapter_outputs * routing_weights).sum(dim=0)
        
        return weighted_output
    
    def get_total_params(self) -> int:
        """Get total number of parameters across all adapters."""
        return sum(adapter.get_num_params() for adapter in self.adapters)


# ============================================================================
# ROUTING NETWORK
# ============================================================================

class TypologyRouter(nn.Module):
    """
    Routes to K adapters based on typological embeddings.
    Generates a probability distribution over adapters for each input.
    """
    
    def __init__(
        self,
        typology_dim: int = 128,
        num_adapters: int = 10,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        """
        Args:
            typology_dim: Dimension of typological embeddings
            num_adapters: Number of adapters to route to
            hidden_dim: Hidden dimension of the routing network
            temperature: Temperature for softmax (higher = more uniform)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_adapters = num_adapters
        self.temperature = temperature
        
        self.router = nn.Sequential(
            nn.Linear(typology_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_adapters)
        )
    
    def forward(self, typology_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate routing weights from typological embeddings.
        
        Args:
            typology_embedding: (batch_size, typology_dim)
        
        Returns:
            routing_weights: (batch_size, num_adapters) - softmax distribution
        """
        logits = self.router(typology_embedding)
        routing_weights = torch.softmax(logits / self.temperature, dim=-1)
        return routing_weights


# ============================================================================
# COMPLETE TADR MODEL
# ============================================================================

class TADRModel(nn.Module):
    """
    Complete TADR (Typology-Aware Dynamic Routing) Model.
    
    Integrates:
        - Step 1: Typological feature embeddings
        - Step 2: Multilingual base model (XLM-R/mBERT)
        - Step 3: Multi-adapter modules with dynamic routing
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        feature_file: str = 'wals_features.csv',
        # Typology Module Config
        typology_embedding_dim: int = 128,
        typology_hidden_dim: int = 256,
        typology_dropout: float = 0.1,
        # Adapter Config
        num_adapters: int = 10,
        adapter_bottleneck_size: int = 64,
        adapter_non_linearity: str = "relu",
        adapter_dropout: float = 0.1,
        num_adapter_layers: Optional[int] = None,
        # Router Config
        router_hidden_dim: int = 64,
        router_temperature: float = 1.0,
        router_dropout: float = 0.1,
        # General Config
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            feature_file: Path to typological features CSV
            typology_embedding_dim: Output dimension of typology embeddings
            typology_hidden_dim: Hidden dimension for typology MLP
            typology_dropout: Dropout for typology module
            num_adapters: Number of adapters per layer
            adapter_bottleneck_size: Bottleneck dimension for adapters
            adapter_non_linearity: Activation function for adapters
            adapter_dropout: Dropout for adapters
            num_adapter_layers: Number of layers to add adapters to (None = all)
            router_hidden_dim: Hidden dimension for routing network
            router_temperature: Temperature for routing softmax
            router_dropout: Dropout for router
            device: Device to use
        """
        super().__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*60)
        print("Initializing TADR Model")
        print("="*60)
        
        # ========================
        # Step 1: Typology Module
        # ========================
        feature_loader = TypologicalFeatureLoader(
            feature_file=feature_file,
            missing_value_strategy='mean'
        )
        self.typology_module = TypologyFeatureModule(
            feature_loader=feature_loader,
            embedding_dim=typology_embedding_dim,
            hidden_dim=typology_hidden_dim,
            dropout=typology_dropout
        )
        
        # ========================
        # Step 2: Base Model
        # ========================
        self.base_model_wrapper = ModelWithAdapterSlots(
            model_name=model_name,
            freeze_base=True,
            num_adapter_layers=num_adapter_layers,
            device=self.device
        )
        
        self.config = self.base_model_wrapper.config
        self.hidden_size = self.base_model_wrapper.hidden_size
        self.adapter_layer_indices = self.base_model_wrapper.adapter_layer_indices
        self.num_adapter_layers = len(self.adapter_layer_indices)
        
        # ========================
        # Step 3: Adapters & Router
        # ========================
        
        # Multi-adapter modules (one per transformer layer)
        self.multi_adapters = nn.ModuleList([
            MultiAdapterModule(
                num_adapters=num_adapters,
                hidden_size=self.hidden_size,
                bottleneck_size=adapter_bottleneck_size,
                non_linearity=adapter_non_linearity,
                dropout=adapter_dropout
            )
            for _ in range(self.num_adapter_layers)
        ])
        
        # Router network
        self.router = TypologyRouter(
            typology_dim=typology_embedding_dim,
            num_adapters=num_adapters,
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
            dropout=router_dropout
        )
        
        # Move to device
        self.to(self.device)
        
        # Print summary
        self._print_summary(num_adapters, adapter_bottleneck_size)
    
    def _print_summary(self, num_adapters: int, bottleneck_size: int):
        """Print model configuration summary."""
        total_params = self.count_parameters()
        trainable_params = self.count_trainable_parameters()
        adapter_params = self.count_adapter_parameters()
        
        print(f"\n✅ Model Configuration:")
        print(f"  Base model: {self.base_model_wrapper.model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num transformer layers: {self.base_model_wrapper.num_layers}")
        print(f"  Layers with adapters: {self.num_adapter_layers}")
        print(f"  Adapters per layer: {num_adapters}")
        print(f"  Adapter bottleneck: {bottleneck_size}")
        
        print(f"\n📊 Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Adapters only: {adapter_params:,} ({adapter_params/total_params*100:.4f}%)")
        print("="*60 + "\n")
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_adapter_parameters(self) -> int:
        """Count adapter and router parameters."""
        adapter_params = sum(p.numel() for p in self.multi_adapters.parameters())
        router_params = sum(p.numel() for p in self.router.parameters())
        return adapter_params + router_params
    
    def forward(
        self,
        lang_ids: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_routing_weights: bool = False,
        return_dict: bool = True
    ):
        """
        Forward pass through the complete TADR model.
        
        Args:
            lang_ids: List of language ISO codes for the batch
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs (batch_size, seq_len)
            return_routing_weights: Whether to return routing weights
            return_dict: Whether to return as dict or namedtuple
        
        Returns:
            TADROutput or dict with model outputs
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # ========================
        # Get typology embeddings
        # ========================
        typology_embeddings = self.typology_module(lang_ids)
        typology_embeddings = typology_embeddings.to(self.device)
        
        # ========================
        # Get routing weights
        # ========================
        routing_weights = self.router(typology_embeddings)
        
        # ========================
        # Prepare for forward pass
        # ========================
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = self.base_model_wrapper.model.get_extended_attention_mask(
            attention_mask, input_ids.size(), device=self.device
        )
        
        # ========================
        # Embeddings
        # ========================
        hidden_states = self.base_model_wrapper.model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        all_hidden_states = [hidden_states]
        
        # ========================
        # Transformer layers
        # ========================
        encoder_layers = self.base_model_wrapper.encoder_layers
        adapter_idx = 0
        
        for layer_idx, layer_module in enumerate(encoder_layers):
            # Standard transformer layer forward pass
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask
            )
            hidden_states = layer_outputs[0]
            
            # Apply multi-adapter with routing if this layer has adapters
            if layer_idx in self.adapter_layer_indices:
                hidden_states = self.multi_adapters[adapter_idx](
                    hidden_states,
                    routing_weights
                )
                adapter_idx += 1
            
            all_hidden_states.append(hidden_states)
        
        # ========================
        # Pooler
        # ========================
        pooler_output = None
        if hasattr(self.base_model_wrapper.model, 'pooler') and \
           self.base_model_wrapper.model.pooler is not None:
            pooler_output = self.base_model_wrapper.model.pooler(hidden_states)
        else:
            # Use CLS token if no pooler
            pooler_output = hidden_states[:, 0]
        
        # ========================
        # Return outputs
        # ========================
        if return_dict:
            outputs = {
                'last_hidden_state': hidden_states,
                'pooler_output': pooler_output,
                'hidden_states': tuple(all_hidden_states)
            }
            if return_routing_weights:
                outputs['routing_weights'] = routing_weights
            return outputs
        else:
            return TADROutput(
                last_hidden_state=hidden_states,
                pooler_output=pooler_output,
                hidden_states=tuple(all_hidden_states),
                routing_weights=routing_weights if return_routing_weights else None
            )
    
    def get_routing_weights(self, lang_ids: List[str]) -> torch.Tensor:
        """
        Get routing weights for given languages without full forward pass.
        
        Args:
            lang_ids: List of language ISO codes
        
        Returns:
            routing_weights: (batch_size, num_adapters)
        """
        typology_embeddings = self.typology_module(lang_ids)
        typology_embeddings = typology_embeddings.to(self.device)
        routing_weights = self.router(typology_embeddings)
        return routing_weights


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_adapter_params(hidden_size: int, bottleneck_size: int) -> int:
    """
    Calculate number of parameters in a single adapter.
    
    Args:
        hidden_size: Hidden dimension
        bottleneck_size: Bottleneck dimension
    
    Returns:
        Number of parameters
    """
    # Down-projection
    down_params = hidden_size * bottleneck_size + bottleneck_size
    
    # Up-projection
    up_params = bottleneck_size * hidden_size + hidden_size
    
    # Layer norm (gamma and beta)
    ln_params = 2 * hidden_size
    
    return down_params + up_params + ln_params


class AdapterConfig:
    """Configuration class for adapter modules."""
    
    def __init__(
        self,
        num_adapters: int = 10,
        hidden_size: int = 768,
        bottleneck_size: int = 64,
        non_linearity: str = "relu",
        dropout: float = 0.1,
        init_scale: float = 1e-3,
        use_layer_norm: bool = True,
        adapter_names: Optional[List[str]] = None
    ):
        self.num_adapters = num_adapters
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.init_scale = init_scale
        self.use_layer_norm = use_layer_norm
        self.adapter_names = adapter_names
        self.reduction_factor = hidden_size / bottleneck_size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'num_adapters': self.num_adapters,
            'hidden_size': self.hidden_size,
            'bottleneck_size': self.bottleneck_size,
            'non_linearity': self.non_linearity,
            'dropout': self.dropout,
            'init_scale': self.init_scale,
            'use_layer_norm': self.use_layer_norm,
            'adapter_names': self.adapter_names,
            'reduction_factor': self.reduction_factor
        }

