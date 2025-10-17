"""
TADR Framework - Step 4: Dynamic Routing Network
Implementation of the router that combines typology + context to select adapters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class RouterMLP(nn.Module):
    """
    Core MLP router that produces adapter weights.
    
    Input: Concatenation of typology embedding + contextual features
    Output: K routing weights (one per adapter)
    """
    
    def __init__(
        self,
        typology_dim: int = 128,
        context_dim: int = 768,
        num_adapters: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Args:
            typology_dim: Dimension of typology embeddings
            context_dim: Dimension of contextual features (e.g., CLS token)
            num_adapters: Number of adapters (K)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.typology_dim = typology_dim
        self.context_dim = context_dim
        self.num_adapters = num_adapters
        
        # Input dimension
        input_dim = typology_dim + context_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_adapters))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        typology_embedding: torch.Tensor,
        context_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute routing logits.
        
        Args:
            typology_embedding: (batch_size, typology_dim)
            context_features: (batch_size, context_dim)
        
        Returns:
            routing_logits: (batch_size, num_adapters)
        """
        # Concatenate inputs
        combined = torch.cat([typology_embedding, context_features], dim=-1)
        
        # Pass through MLP
        logits = self.mlp(combined)
        
        return logits


class SoftmaxGating(nn.Module):
    """
    Softmax gating mechanism: all adapters used with normalized weights.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature for softmax (higher = more uniform)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_adapters)
        
        Returns:
            weights: (batch_size, num_adapters), sums to 1
        """
        return F.softmax(logits / self.temperature, dim=-1)


class TopKGating(nn.Module):
    """
    Top-K sparse gating: only top K adapters are used.
    Provides sparsity for efficiency.
    """
    
    def __init__(self, k: int = 2, temperature: float = 1.0):
        """
        Args:
            k: Number of adapters to select
            temperature: Temperature for softmax
        """
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_adapters)
        
        Returns:
            weights: (batch_size, num_adapters), sparse with only K non-zero
        """
        batch_size, num_adapters = logits.shape
        
        # Get top-k indices
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Apply softmax only to top-k
        top_k_weights = F.softmax(top_k_logits / self.temperature, dim=-1)
        
        # Create sparse weight tensor
        weights = torch.zeros_like(logits)
        weights.scatter_(1, top_k_indices, top_k_weights)
        
        return weights


class ThresholdGating(nn.Module):
    """
    Threshold gating: use adapters with activation above threshold.
    Adaptive sparsity based on confidence.
    """
    
    def __init__(self, threshold: float = 0.1, temperature: float = 1.0):
        """
        Args:
            threshold: Minimum weight threshold
            temperature: Temperature for softmax
        """
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_adapters)
        
        Returns:
            weights: (batch_size, num_adapters), sparse with adaptive K
        """
        # Apply softmax
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        # Apply threshold mask
        mask = weights >= self.threshold
        
        # Renormalize among selected adapters
        masked_weights = weights * mask.float()
        sum_weights = masked_weights.sum(dim=-1, keepdim=True)
        
        # Avoid division by zero
        sum_weights = torch.where(sum_weights > 0, sum_weights, torch.ones_like(sum_weights))
        
        normalized_weights = masked_weights / sum_weights
        
        return normalized_weights


class DynamicRouter(nn.Module):
    """
    Complete dynamic routing network.
    Combines router MLP with gating mechanism.
    """
    
    def __init__(
        self,
        typology_dim: int = 128,
        context_dim: int = 768,
        num_adapters: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        gating_type: str = "softmax",  # "softmax", "topk", or "threshold"
        gating_config: Optional[Dict] = None
    ):
        """
        Args:
            typology_dim: Dimension of typology embeddings
            context_dim: Dimension of contextual features
            num_adapters: Number of adapters
            hidden_dims: Hidden layer dimensions for MLP
            dropout: Dropout probability
            gating_type: Type of gating mechanism
            gating_config: Configuration dict for gating (e.g., {"k": 2, "temperature": 1.0})
        """
        super().__init__()
        
        self.typology_dim = typology_dim
        self.context_dim = context_dim
        self.num_adapters = num_adapters
        self.gating_type = gating_type
        
        # Router MLP
        self.router_mlp = RouterMLP(
            typology_dim=typology_dim,
            context_dim=context_dim,
            num_adapters=num_adapters,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # Gating mechanism
        gating_config = gating_config or {}
        
        if gating_type == "softmax":
            self.gating = SoftmaxGating(
                temperature=gating_config.get("temperature", 1.0)
            )
        elif gating_type == "topk":
            self.gating = TopKGating(
                k=gating_config.get("k", 2),
                temperature=gating_config.get("temperature", 1.0)
            )
        elif gating_type == "threshold":
            self.gating = ThresholdGating(
                threshold=gating_config.get("threshold", 0.1),
                temperature=gating_config.get("temperature", 1.0)
            )
        else:
            raise ValueError(f"Unknown gating type: {gating_type}")
    
    def forward(
        self,
        typology_embedding: torch.Tensor,
        context_features: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute routing weights.
        
        Args:
            typology_embedding: (batch_size, typology_dim)
            context_features: (batch_size, context_dim)
            return_logits: Whether to return raw logits
        
        Returns:
            routing_weights: (batch_size, num_adapters)
            logits: (batch_size, num_adapters) if return_logits=True
        """
        # Get logits from router MLP
        logits = self.router_mlp(typology_embedding, context_features)
        
        # Apply gating
        weights = self.gating(logits)
        
        if return_logits:
            return weights, logits
        return weights, None
    
    def get_routing_distribution(
        self,
        typology_embedding: torch.Tensor,
        context_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed routing information for analysis.
        
        Returns:
            Dictionary with weights, logits, entropy, etc.
        """
        weights, logits = self.forward(
            typology_embedding, 
            context_features, 
            return_logits=True
        )
        
        # Compute entropy (measure of uncertainty)
        # H = -sum(p * log(p))
        epsilon = 1e-10
        entropy = -(weights * torch.log(weights + epsilon)).sum(dim=-1)
        
        # Compute sparsity (number of active adapters)
        sparsity = (weights > 1e-5).sum(dim=-1).float()
        
        # Get top adapter indices
        top_indices = torch.argmax(weights, dim=-1)
        
        return {
            'weights': weights,
            'logits': logits,
            'entropy': entropy,
            'sparsity': sparsity,
            'top_indices': top_indices,
            'max_weight': weights.max(dim=-1)[0]
        }


class ContextExtractor(nn.Module):
    """
    Extracts contextual features from hidden states.
    Multiple strategies: CLS token, mean pooling, attention pooling.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        pooling_type: str = "cls",  # "cls", "mean", or "attention"
    ):
        """
        Args:
            hidden_size: Hidden dimension
            pooling_type: Type of pooling strategy
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            # Learnable attention pooling
            self.attention_weights = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract context features.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            context_features: (batch_size, hidden_size)
        """
        if self.pooling_type == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_type == "mean":
            # Mean pooling over sequence
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask
            else:
                return hidden_states.mean(dim=1)
        
        elif self.pooling_type == "attention":
            # Attention-weighted pooling
            attn_scores = self.attention_weights(hidden_states).squeeze(-1)  # (batch, seq_len)
            
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
            weighted = (hidden_states * attn_weights).sum(dim=1)  # (batch, hidden_size)
            
            return weighted
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss to encourage uniform adapter usage.
    Prevents router from collapsing to single adapter.
    """
    
    def __init__(self, num_adapters: int, alpha: float = 0.01):
        """
        Args:
            num_adapters: Number of adapters
            alpha: Weight for load balancing loss
        """
        super().__init__()
        self.num_adapters = num_adapters
        self.alpha = alpha
    
    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            routing_weights: (batch_size, num_adapters)
        
        Returns:
            loss: Scalar load balancing loss
        """
        # Average usage across batch
        avg_usage = routing_weights.mean(dim=0)  # (num_adapters,)
        
        # Target uniform distribution
        target = torch.ones_like(avg_usage) / self.num_adapters
        
        # L2 loss between actual and target distribution
        loss = F.mse_loss(avg_usage, target)
        
        return self.alpha * loss


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def test_router_mlp():
    """Test the core router MLP."""
    print("="*60)
    print("Testing Router MLP")
    print("="*60)
    
    router = RouterMLP(
        typology_dim=128,
        context_dim=768,
        num_adapters=10,
        hidden_dims=[256, 128]
    )
    
    print(f"\nRouter Configuration:")
    print(f"  Input: typology(128) + context(768) = 896")
    print(f"  Hidden layers: {[256, 128]}")
    print(f"  Output: {10} adapter logits")
    print(f"  Parameters: {sum(p.numel() for p in router.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    typology_emb = torch.randn(batch_size, 128)
    context_feat = torch.randn(batch_size, 768)
    
    logits = router(typology_emb, context_feat)
    
    print(f"\nForward Pass:")
    print(f"  Input shapes: ({batch_size}, 128) + ({batch_size}, 768)")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Sample logits: {logits[0]}")
    
    print("\n✅ Router MLP test passed!")


def test_gating_mechanisms():
    """Test different gating mechanisms."""
    print("\n" + "="*60)
    print("Testing Gating Mechanisms")
    print("="*60)
    
    batch_size = 4
    num_adapters = 10
    logits = torch.randn(batch_size, num_adapters)
    
    print(f"\nInput logits shape: {logits.shape}")
    print(f"Sample logits: {logits[0]}")
    
    # Test 1: Softmax gating
    print("\n1. Softmax Gating:")
    softmax_gate = SoftmaxGating(temperature=1.0)
    weights = softmax_gate(logits)
    print(f"   Weights: {weights[0]}")
    print(f"   Sum: {weights[0].sum():.4f}")
    print(f"   Active adapters: {(weights[0] > 0.01).sum().item()}")
    
    # Test 2: Top-K gating
    print("\n2. Top-K Gating (k=2):")
    topk_gate = TopKGating(k=2, temperature=1.0)
    weights = topk_gate(logits)
    print(f"   Weights: {weights[0]}")
    print(f"   Sum: {weights[0].sum():.4f}")
    print(f"   Active adapters: {(weights[0] > 0.01).sum().item()}")
    
    # Test 3: Threshold gating
    print("\n3. Threshold Gating (threshold=0.15):")
    threshold_gate = ThresholdGating(threshold=0.15, temperature=1.0)
    weights = threshold_gate(logits)
    print(f"   Weights: {weights[0]}")
    print(f"   Sum: {weights[0].sum():.4f}")
    print(f"   Active adapters: {(weights[0] > 0.01).sum().item()}")
    
    print("\n✅ Gating mechanisms test passed!")


def test_dynamic_router():
    """Test complete dynamic router."""
    print("\n" + "="*60)
    print("Testing Dynamic Router")
    print("="*60)
    
    # Create router
    router = DynamicRouter(
        typology_dim=128,
        context_dim=768,
        num_adapters=10,
        hidden_dims=[256, 128],
        gating_type="topk",
        gating_config={"k": 3, "temperature": 1.0}
    )
    
    print(f"\nRouter Configuration:")
    print(f"  Gating type: {router.gating_type}")
    print(f"  Number of adapters: {router.num_adapters}")
    
    # Test forward pass
    batch_size = 4
    typology_emb = torch.randn(batch_size, 128)
    context_feat = torch.randn(batch_size, 768)
    
    weights, logits = router(typology_emb, context_feat, return_logits=True)
    
    print(f"\nForward Pass:")
    print(f"  Routing weights shape: {weights.shape}")
    print(f"  Sample weights: {weights[0]}")
    print(f"  Sparsity: {(weights[0] > 0).sum().item()} active adapters")
    
    # Test routing distribution
    print("\n Detailed Routing Analysis:")
    routing_info = router.get_routing_distribution(typology_emb, context_feat)
    
    print(f"  Entropy: {routing_info['entropy'][0]:.4f}")
    print(f"  Sparsity: {routing_info['sparsity'][0]:.0f} adapters")
    print(f"  Top adapter: {routing_info['top_indices'][0].item()}")
    print(f"  Max weight: {routing_info['max_weight'][0]:.4f}")
    
    print("\n✅ Dynamic router test passed!")


def test_context_extractor():
    """Test context extraction strategies."""
    print("\n" + "="*60)
    print("Testing Context Extractor")
    print("="*60)
    
    batch_size, seq_len, hidden_size = 4, 128, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 100:] = 0  # Mask last 28 tokens
    
    strategies = ["cls", "mean", "attention"]
    
    for strategy in strategies:
        extractor = ContextExtractor(hidden_size, pooling_type=strategy)
        context = extractor(hidden_states, attention_mask)
        
        print(f"\n{strategy.upper()} Pooling:")
        print(f"  Output shape: {context.shape}")
        print(f"  Output mean: {context.mean().item():.4f}")
        print(f"  Output std: {context.std().item():.4f}")
    
    print("\n✅ Context extractor test passed!")


def test_load_balancing_loss():
    """Test load balancing loss."""
    print("\n" + "="*60)
    print("Testing Load Balancing Loss")
    print("="*60)
    
    lb_loss = LoadBalancingLoss(num_adapters=10, alpha=0.01)
    
    # Test 1: Balanced distribution
    print("\n1. Balanced routing:")
    balanced = torch.ones(32, 10) / 10
    loss = lb_loss(balanced)
    print(f"   Loss: {loss.item():.6f}")
    
    # Test 2: Imbalanced (collapsed to one adapter)
    print("\n2. Collapsed routing:")
    collapsed = torch.zeros(32, 10)
    collapsed[:, 0] = 1.0
    loss = lb_loss(collapsed)
    print(f"   Loss: {loss.item():.6f}")
    
    # Test 3: Partially imbalanced
    print("\n3. Partially imbalanced:")
    partial = F.softmax(torch.randn(32, 10) * 2, dim=-1)
    loss = lb_loss(partial)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Adapter usage: {partial.mean(dim=0)}")
    
    print("\n✅ Load balancing loss test passed!")


def demo_full_routing_pipeline():
    """Demonstrate complete routing pipeline."""
    print("\n" + "="*60)
    print("Full Routing Pipeline Demo")
    print("="*60)
    
    print("\nPipeline Steps:")
    print("1. Extract context from hidden states")
    print("2. Get typology embedding for language")
    print("3. Router produces weights")
    print("4. Apply weighted adapters")
    print()
    
    # Initialize components
    context_extractor = ContextExtractor(768, pooling_type="cls")
    router = DynamicRouter(
        typology_dim=128,
        context_dim=768,
        num_adapters=10,
        gating_type="softmax"
    )
    
    # Simulate data
    batch_size = 2
    hidden_states = torch.randn(batch_size, 128, 768)
    typology_emb = torch.randn(batch_size, 128)
    
    # Step 1: Extract context
    context_features = context_extractor(hidden_states)
    print(f"✓ Context extracted: {context_features.shape}")
    
    # Step 2: Get routing weights
    routing_weights, _ = router(typology_emb, context_features)
    print(f"✓ Routing weights computed: {routing_weights.shape}")
    print(f"  Sample weights: {routing_weights[0]}")
    
    # Step 3: Would apply to adapters (see Step 5 integration)
    print(f"✓ Ready to apply to MultiAdapterModule")
    
    print("\n✅ Full pipeline demo complete!")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("TADR Framework - Step 4: Dynamic Routing Network")
    print("🚀" * 30)
    
    # Run all tests
    test_router_mlp()
    test_gating_mechanisms()
    test_dynamic_router()
    test_context_extractor()
    test_load_balancing_loss()
    demo_full_routing_pipeline()
    
    print("\n" + "="*60)
    print("✅ Step 4 Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  → Step 5: Integration Layer")
    print("  → Combine all components into full TADR model")
    print("="*60 + "\n")