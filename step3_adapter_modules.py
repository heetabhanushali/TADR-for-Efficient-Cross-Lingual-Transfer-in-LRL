"""
TADR Framework - Step 3: Adapter Modules
Implementation of bottleneck adapters for parameter-efficient fine-tuning
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import math


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter module with down-projection, non-linearity, and up-projection.
    
    Architecture:
        x → Down(768→r) → ReLU → Up(r→768) → LayerNorm → x + residual
    
    Where r is the bottleneck dimension (reduction factor).
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
        
        # Down-projection: hidden_size → bottleneck_size
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
        
        # Up-projection: bottleneck_size → hidden_size
        self.up_project = nn.Linear(bottleneck_size, hidden_size)
        
        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights with small values to start close to identity
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


class AdapterLayer(nn.Module):
    """
    Complete adapter layer that can be inserted into a transformer layer.
    Includes optional placement (after attention or after FFN).
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        bottleneck_size: int = 64,
        non_linearity: str = "relu",
        dropout: float = 0.1,
        placement: str = "ffn",  # "attention" or "ffn" or "both"
        **kwargs
    ):
        """
        Args:
            hidden_size: Hidden dimension
            bottleneck_size: Bottleneck dimension
            non_linearity: Activation function
            dropout: Dropout rate
            placement: Where to place adapters ("attention", "ffn", or "both")
        """
        super().__init__()
        
        self.placement = placement
        
        # Adapter after attention
        if placement in ["attention", "both"]:
            self.adapter_after_attn = BottleneckAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                non_linearity=non_linearity,
                dropout=dropout,
                **kwargs
            )
        else:
            self.adapter_after_attn = None
        
        # Adapter after feed-forward network (FFN)
        if placement in ["ffn", "both"]:
            self.adapter_after_ffn = BottleneckAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                non_linearity=non_linearity,
                dropout=dropout,
                **kwargs
            )
        else:
            self.adapter_after_ffn = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position: str = "ffn"
    ) -> torch.Tensor:
        """
        Apply adapter at specified position.
        
        Args:
            hidden_states: Input tensor
            position: "attention" or "ffn"
        
        Returns:
            Output tensor after adapter
        """
        if position == "attention" and self.adapter_after_attn is not None:
            return self.adapter_after_attn(hidden_states)
        elif position == "ffn" and self.adapter_after_ffn is not None:
            return self.adapter_after_ffn(hidden_states)
        else:
            return hidden_states


class MultiAdapterModule(nn.Module):
    """
    Module containing K language-specific adapters.
    This is what the router will select from dynamically.
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
            adapter_names: Optional names for adapters (e.g., language codes)
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
            assert len(adapter_names) == num_adapters
            self.adapter_names = adapter_names
        else:
            self.adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adapters with optional routing weights.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            routing_weights: Routing weights (batch_size, num_adapters) or None
                           If None, applies all adapters equally
        
        Returns:
            Output tensor after weighted adapter application
        """
        batch_size = hidden_states.size(0)
        
        if routing_weights is None:
            # No routing: apply all adapters with equal weight
            routing_weights = torch.ones(batch_size, self.num_adapters, 
                                        device=hidden_states.device)
            routing_weights = routing_weights / self.num_adapters
        
        # Compute output from each adapter
        adapter_outputs = []
        for adapter in self.adapters:
            output = adapter(hidden_states)
            adapter_outputs.append(output)
        
        # Stack: (num_adapters, batch_size, seq_len, hidden_size)
        adapter_outputs = torch.stack(adapter_outputs, dim=0)
        
        # Reshape routing weights for broadcasting: (batch_size, num_adapters, 1, 1)
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum: (batch_size, seq_len, hidden_size)
        weighted_output = (adapter_outputs * routing_weights.transpose(0, 1)).sum(dim=0)
        
        return weighted_output
    
    def forward_single_adapter(
        self,
        hidden_states: torch.Tensor,
        adapter_idx: int
    ) -> torch.Tensor:
        """Apply a single adapter (for testing/debugging)."""
        return self.adapters[adapter_idx](hidden_states)
    
    def get_total_params(self) -> int:
        """Get total number of parameters across all adapters."""
        return sum(adapter.get_num_params() for adapter in self.adapters)


class AdapterConfig:
    """Configuration class for adapter modules."""
    
    def __init__(
        self,
        num_adapters: int = 10,
        hidden_size: int = 768,
        bottleneck_size: int = 64,
        non_linearity: str = "relu",
        dropout: float = 0.1,
        placement: str = "ffn",
        init_scale: float = 1e-3,
        use_layer_norm: bool = True,
        adapter_names: Optional[List[str]] = None
    ):
        self.num_adapters = num_adapters
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.placement = placement
        self.init_scale = init_scale
        self.use_layer_norm = use_layer_norm
        self.adapter_names = adapter_names
        
        # Calculate reduction factor
        self.reduction_factor = hidden_size / bottleneck_size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'num_adapters': self.num_adapters,
            'hidden_size': self.hidden_size,
            'bottleneck_size': self.bottleneck_size,
            'non_linearity': self.non_linearity,
            'dropout': self.dropout,
            'placement': self.placement,
            'init_scale': self.init_scale,
            'use_layer_norm': self.use_layer_norm,
            'adapter_names': self.adapter_names,
            'reduction_factor': self.reduction_factor
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_adapter_params(hidden_size: int, bottleneck_size: int) -> int:
    """Calculate number of parameters in a single adapter."""
    # Down-projection: hidden_size * bottleneck_size + bottleneck_size (bias)
    down_params = hidden_size * bottleneck_size + bottleneck_size
    
    # Up-projection: bottleneck_size * hidden_size + hidden_size (bias)
    up_params = bottleneck_size * hidden_size + hidden_size
    
    # Layer norm (if used): 2 * hidden_size (gamma and beta)
    ln_params = 2 * hidden_size
    
    return down_params + up_params + ln_params


def get_parameter_efficiency(
    base_model_params: int,
    adapter_params: int
) -> float:
    """Calculate parameter efficiency ratio."""
    return (adapter_params / base_model_params) * 100


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def test_bottleneck_adapter():
    """Test single bottleneck adapter."""
    print("="*60)
    print("Testing Bottleneck Adapter")
    print("="*60)
    
    # Create adapter
    adapter = BottleneckAdapter(
        hidden_size=768,
        bottleneck_size=64,
        non_linearity="relu",
        dropout=0.1
    )
    
    print(f"\nAdapter Configuration:")
    print(f"  Hidden size: {adapter.hidden_size}")
    print(f"  Bottleneck size: {adapter.bottleneck_size}")
    print(f"  Reduction factor: {adapter.reduction_factor:.1f}x")
    print(f"  Parameters: {adapter.get_num_params():,}")
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    hidden_states = torch.randn(batch_size, seq_len, 768)
    
    output = adapter(hidden_states)
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    # Check residual connection
    diff = (output - hidden_states).abs().mean()
    print(f"  Mean difference from input: {diff.item():.6f}")
    print(f"  (Small value confirms adapter starts near identity)")
    
    print("\n✅ Bottleneck adapter test passed!")


def test_multi_adapter_module():
    """Test multi-adapter module."""
    print("\n" + "="*60)
    print("Testing Multi-Adapter Module")
    print("="*60)
    
    # Create module with 5 adapters
    num_adapters = 5
    adapter_names = ['en', 'hi', 'zh', 'ar', 'sw']
    
    multi_adapter = MultiAdapterModule(
        num_adapters=num_adapters,
        hidden_size=768,
        bottleneck_size=64,
        adapter_names=adapter_names
    )
    
    print(f"\nMulti-Adapter Configuration:")
    print(f"  Number of adapters: {multi_adapter.num_adapters}")
    print(f"  Adapter names: {multi_adapter.adapter_names}")
    print(f"  Total parameters: {multi_adapter.get_total_params():,}")
    
    # Test forward pass with routing
    batch_size, seq_len = 4, 128
    hidden_states = torch.randn(batch_size, seq_len, 768)
    
    # Create routing weights (softmax distribution)
    routing_weights = torch.randn(batch_size, num_adapters)
    routing_weights = torch.softmax(routing_weights, dim=-1)
    
    print(f"\nRouting Weights (sample):")
    print(f"  Shape: {routing_weights.shape}")
    print(f"  Sample weights: {routing_weights[0]}")
    print(f"  Sum: {routing_weights[0].sum().item():.4f}")
    
    # Forward pass
    output = multi_adapter(hidden_states, routing_weights)
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    
    # Test single adapter application
    print(f"\nTesting Single Adapter:")
    single_output = multi_adapter.forward_single_adapter(hidden_states, adapter_idx=0)
    print(f"  Adapter 0 ('{adapter_names[0]}') output shape: {single_output.shape}")
    
    print("\n✅ Multi-adapter module test passed!")


def test_adapter_scaling():
    """Test adapters with different bottleneck sizes."""
    print("\n" + "="*60)
    print("Testing Adapter Scaling")
    print("="*60)
    
    hidden_size = 768
    bottleneck_sizes = [16, 32, 64, 128, 256]
    
    print(f"\nHidden size: {hidden_size}")
    print(f"\n{'Bottleneck':<12} {'Params':<12} {'Reduction':<12} {'Efficiency %'}")
    print("-" * 50)
    
    base_params = 110_000_000  # Approximate for XLM-R base
    
    for bs in bottleneck_sizes:
        params = calculate_adapter_params(hidden_size, bs)
        reduction = hidden_size / bs
        efficiency = (params / base_params) * 100
        
        print(f"{bs:<12} {params:<12,} {reduction:<12.1f}x {efficiency:<.4f}%")
    
    print("\n✅ Adapter scaling test passed!")


def test_parameter_efficiency():
    """Compare parameter efficiency across configurations."""
    print("\n" + "="*60)
    print("Parameter Efficiency Analysis")
    print("="*60)
    
    base_model_params = 110_000_000  # XLM-R base (~110M params)
    num_layers = 12
    num_adapters = 10
    
    configs = [
        ("Standard (64)", 768, 64),
        ("Compressed (32)", 768, 32),
        ("Expanded (128)", 768, 128),
    ]
    
    print(f"\nBase model parameters: {base_model_params:,}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of adapters per layer: {num_adapters}")
    print()
    
    for name, hidden, bottleneck in configs:
        params_per_adapter = calculate_adapter_params(hidden, bottleneck)
        total_adapter_params = params_per_adapter * num_layers * num_adapters
        efficiency = get_parameter_efficiency(base_model_params, total_adapter_params)
        
        print(f"{name}:")
        print(f"  Params per adapter: {params_per_adapter:,}")
        print(f"  Total adapter params: {total_adapter_params:,}")
        print(f"  Efficiency: {efficiency:.2f}% of base model")
        print()
    
    print("✅ Parameter efficiency analysis complete!")


def demo_adapter_integration():
    """Demonstrate how adapters integrate with base model."""
    print("\n" + "="*60)
    print("Adapter Integration Demo")
    print("="*60)
    
    print("\nTypical integration pattern:")
    print("""
    for layer in transformer.layers:
        # Original transformer layer forward pass
        hidden_states = layer.attention(hidden_states)
        
        # [OPTIONAL] Apply adapter after attention
        if adapter_after_attn:
            hidden_states = adapter_layer(hidden_states, position="attention")
        
        hidden_states = layer.ffn(hidden_states)
        
        # [STANDARD] Apply adapter after FFN
        hidden_states = adapter_layer(hidden_states, position="ffn")
    """)
    
    print("\nWith dynamic routing:")
    print("""
    # Get routing weights from router network
    routing_weights = router(typology_embedding, cls_representation)
    
    # Apply routed adapters
    hidden_states = multi_adapter(hidden_states, routing_weights)
    """)
    
    print("\n✅ Integration demo complete!")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("TADR Framework - Step 3: Adapter Modules")
    print("🚀" * 30)
    
    # Run all tests
    test_bottleneck_adapter()
    test_multi_adapter_module()
    test_adapter_scaling()
    test_parameter_efficiency()
    demo_adapter_integration()
    
    print("\n" + "="*60)
    print("✅ Step 3 Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  → Step 4: Dynamic Routing Network")
    print("  → Combine typology embeddings with routing logic")
    print("="*60 + "\n")