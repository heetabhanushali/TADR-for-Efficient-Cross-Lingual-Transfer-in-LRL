"""
TADR Framework - Step 4: Dynamic Routing Network
Implementation of the router that combines typology + context to select adapters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


# ============================================================================
# CORE ROUTER COMPONENTS
# ============================================================================

class RouterMLP(nn.Module):
    """
    Core MLP router that produces adapter weights.
    
    Input: Concatenation of typology embedding + contextual features
    Output: K routing logits (one per adapter)
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
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.typology_dim = typology_dim
        self.context_dim = context_dim
        self.num_adapters = num_adapters
        
        # Input dimension: concatenated typology + context
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


# ============================================================================
# GATING MECHANISMS
# ============================================================================

class SoftmaxGating(nn.Module):
    """
    Softmax gating mechanism: all adapters used with normalized weights.
    Standard choice for dense routing.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature for softmax (higher = more uniform distribution)
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
    Provides sparsity for computational efficiency.
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
            weights: (batch_size, num_adapters), sparse with only K non-zero values
        """
        batch_size, num_adapters = logits.shape
        
        # Get top-k indices and values
        top_k_logits, top_k_indices = torch.topk(logits, min(self.k, num_adapters), dim=-1)
        
        # Apply softmax only to top-k
        top_k_weights = F.softmax(top_k_logits / self.temperature, dim=-1)
        
        # Create sparse weight tensor
        weights = torch.zeros_like(logits)
        weights.scatter_(1, top_k_indices, top_k_weights)
        
        return weights


class ThresholdGating(nn.Module):
    """
    Threshold gating: use adapters with activation above threshold.
    Provides adaptive sparsity based on routing confidence.
    """
    
    def __init__(self, threshold: float = 0.1, temperature: float = 1.0):
        """
        Args:
            threshold: Minimum weight threshold (0-1)
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


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

class ContextExtractor(nn.Module):
    """
    Extracts contextual features from hidden states.
    Supports multiple pooling strategies: CLS token, mean pooling, attention pooling.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        pooling_type: str = "cls"
    ):
        """
        Args:
            hidden_size: Hidden dimension of the model
            pooling_type: Type of pooling ('cls', 'mean', or 'attention')
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
        Extract context features from hidden states.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), optional
        
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


# ============================================================================
# DYNAMIC ROUTER
# ============================================================================

class DynamicRouter(nn.Module):
    """
    Complete dynamic routing network.
    Combines router MLP with gating mechanism to produce adapter routing weights.
    """
    
    def __init__(
        self,
        typology_dim: int = 128,
        context_dim: int = 768,
        num_adapters: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        gating_type: str = "softmax",
        gating_config: Optional[Dict] = None
    ):
        """
        Args:
            typology_dim: Dimension of typology embeddings
            context_dim: Dimension of contextual features
            num_adapters: Number of adapters to route to
            hidden_dims: Hidden layer dimensions for MLP
            dropout: Dropout probability
            gating_type: Type of gating ('softmax', 'topk', or 'threshold')
            gating_config: Configuration dict for gating mechanism
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
            logits: (batch_size, num_adapters) if return_logits=True, else None
        """
        # Get logits from router MLP
        logits = self.router_mlp(typology_embedding, context_features)
        
        # Apply gating mechanism
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
        Get detailed routing information for analysis and debugging.
        
        Args:
            typology_embedding: (batch_size, typology_dim)
            context_features: (batch_size, context_dim)
        
        Returns:
            Dictionary containing:
                - weights: Routing weights
                - logits: Raw logits
                - entropy: Routing entropy (uncertainty measure)
                - sparsity: Number of active adapters
                - top_indices: Index of top adapter
                - max_weight: Maximum routing weight
        """
        weights, logits = self.forward(
            typology_embedding, 
            context_features, 
            return_logits=True
        )
        
        # Compute entropy (measure of routing uncertainty)
        # H = -sum(p * log(p))
        epsilon = 1e-10
        entropy = -(weights * torch.log(weights + epsilon)).sum(dim=-1)
        
        # Compute sparsity (number of active adapters)
        sparsity = (weights > 1e-5).sum(dim=-1).float()
        
        # Get top adapter indices
        top_indices = torch.argmax(weights, dim=-1)
        
        # Get maximum weight
        max_weight = weights.max(dim=-1)[0]
        
        return {
            'weights': weights,
            'logits': logits,
            'entropy': entropy,
            'sparsity': sparsity,
            'top_indices': top_indices,
            'max_weight': max_weight
        }


# ============================================================================
# AUXILIARY LOSS FUNCTIONS
# ============================================================================

class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss to encourage uniform adapter usage across the dataset.
    Prevents router from collapsing to use only a single adapter.
    
    Based on the auxiliary loss in Switch Transformers (Fedus et al., 2021).
    """
    
    def __init__(self, num_adapters: int, alpha: float = 0.01):
        """
        Args:
            num_adapters: Number of adapters
            alpha: Weight coefficient for load balancing loss
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
        
        # Target: uniform distribution
        target = torch.ones_like(avg_usage) / self.num_adapters
        
        # L2 loss between actual and target distribution
        loss = F.mse_loss(avg_usage, target)
        
        return self.alpha * loss


class RouterZLoss(nn.Module):
    """
    Router Z-loss to encourage smaller logits and improve training stability.
    
    Based on the approach in ST-MoE (Zoph et al., 2022).
    """
    
    def __init__(self, alpha: float = 0.001):
        """
        Args:
            alpha: Weight coefficient for Z-loss
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute router Z-loss.
        
        Args:
            logits: (batch_size, num_adapters)
        
        Returns:
            loss: Scalar Z-loss
        """
        # Encourages smaller logit values
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        
        return self.alpha * z_loss


# ============================================================================
# CONFIGURATION
# ============================================================================

class RouterConfig:
    """Configuration class for dynamic router."""
    
    def __init__(
        self,
        typology_dim: int = 128,
        context_dim: int = 768,
        num_adapters: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        gating_type: str = "softmax",
        gating_config: Optional[Dict] = None,
        pooling_type: str = "cls",
        use_load_balancing: bool = True,
        load_balancing_alpha: float = 0.01,
        use_z_loss: bool = False,
        z_loss_alpha: float = 0.001
    ):
        """
        Complete configuration for the routing system.
        
        Args:
            typology_dim: Dimension of typology embeddings
            context_dim: Dimension of contextual features
            num_adapters: Number of adapters
            hidden_dims: Hidden dimensions for router MLP
            dropout: Dropout probability
            gating_type: Type of gating mechanism
            gating_config: Gating-specific configuration
            pooling_type: Context pooling strategy
            use_load_balancing: Whether to use load balancing loss
            load_balancing_alpha: Weight for load balancing loss
            use_z_loss: Whether to use router Z-loss
            z_loss_alpha: Weight for Z-loss
        """
        self.typology_dim = typology_dim
        self.context_dim = context_dim
        self.num_adapters = num_adapters
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.gating_type = gating_type
        self.gating_config = gating_config or {}
        self.pooling_type = pooling_type
        self.use_load_balancing = use_load_balancing
        self.load_balancing_alpha = load_balancing_alpha
        self.use_z_loss = use_z_loss
        self.z_loss_alpha = z_loss_alpha
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'typology_dim': self.typology_dim,
            'context_dim': self.context_dim,
            'num_adapters': self.num_adapters,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'gating_type': self.gating_type,
            'gating_config': self.gating_config,
            'pooling_type': self.pooling_type,
            'use_load_balancing': self.use_load_balancing,
            'load_balancing_alpha': self.load_balancing_alpha,
            'use_z_loss': self.use_z_loss,
            'z_loss_alpha': self.z_loss_alpha
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RouterConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

