"""
TADR Framework - Step 6: Loss Functions
Implementation of task losses, regularization, and load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


# ============================================================================
# TASK LOSS
# ============================================================================

class TaskLoss(nn.Module):
    """
    Standard task-specific losses (classification, sequence labeling, etc.)
    """
    
    def __init__(self, task_type: str = "classification", num_classes: Optional[int] = None):
        """
        Args:
            task_type: Type of task ("classification", "sequence_labeling", "qa")
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif task_type == "sequence_labeling":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute task loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes) or (batch_size, seq_len, num_classes)
            labels: Ground truth labels (batch_size,) or (batch_size, seq_len)
            attention_mask: Attention mask for sequence labeling
        
        Returns:
            loss: Scalar loss value
        """
        if self.task_type == "classification":
            return self.loss_fn(logits, labels)
        
        elif self.task_type == "sequence_labeling":
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            return self.loss_fn(logits_flat, labels_flat)


# ============================================================================
# LOAD BALANCING LOSS
# ============================================================================

class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss to encourage uniform adapter usage across the batch.
    Prevents router collapse (using only one adapter for all inputs).
    """
    
    def __init__(
        self,
        num_adapters: int,
        alpha: float = 0.01,
        loss_type: str = "mse"
    ):
        """
        Args:
            num_adapters: Number of adapters (K)
            alpha: Weight for load balancing loss
            loss_type: Type of load balancing loss ("mse", "entropy", "auxiliary")
        """
        super().__init__()
        self.num_adapters = num_adapters
        self.alpha = alpha
        self.loss_type = loss_type
    
    def forward(
        self,
        routing_weights: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            routing_weights: Routing weights (batch_size, num_adapters) or list of tensors
            layer_idx: Layer index (for logging/debugging)
        
        Returns:
            loss: Scalar load balancing loss
        """
        if isinstance(routing_weights, list):
            losses = [self._compute_single_loss(w) for w in routing_weights]
            return sum(losses) / len(losses)
        else:
            return self._compute_single_loss(routing_weights)
    
    def _compute_single_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute loss for a single set of routing weights."""
        avg_usage = routing_weights.mean(dim=0)
        
        if self.loss_type == "mse":
            target = torch.ones_like(avg_usage) / self.num_adapters
            loss = F.mse_loss(avg_usage, target)
        
        elif self.loss_type == "entropy":
            epsilon = 1e-10
            entropy = -(avg_usage * torch.log(avg_usage + epsilon)).sum()
            max_entropy = np.log(self.num_adapters)
            loss = (max_entropy - entropy) / max_entropy
        
        elif self.loss_type == "auxiliary":
            f_i = avg_usage
            P_i = routing_weights.mean(dim=0)
            loss = self.num_adapters * (f_i * P_i).sum()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.alpha * loss


# ============================================================================
# TYPOLOGY REGULARIZATION LOSS
# ============================================================================

class TypologyRegularizationLoss(nn.Module):
    """
    Typology-aware regularization loss.
    Encourages similar adapter usage for typologically similar languages.
    """
    
    def __init__(
        self,
        num_adapters: int,
        alpha: float = 0.01,
        similarity_temperature: float = 1.0
    ):
        """
        Args:
            num_adapters: Number of adapters
            alpha: Weight for regularization loss
            similarity_temperature: Temperature for similarity computation
        """
        super().__init__()
        self.num_adapters = num_adapters
        self.alpha = alpha
        self.temperature = similarity_temperature
    
    def forward(
        self,
        routing_weights: torch.Tensor,
        typology_embeddings: torch.Tensor,
        language_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> torch.Tensor:
        """
        Compute typology regularization loss.
        
        Args:
            routing_weights: Routing weights (batch_size, num_adapters)
            typology_embeddings: Typology embeddings (batch_size, typology_dim)
            language_pairs: Optional pairs of indices for contrastive learning
        
        Returns:
            loss: Scalar regularization loss
        """
        batch_size = routing_weights.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=routing_weights.device)
        
        # Compute pairwise typological similarity
        typology_sim = F.cosine_similarity(
            typology_embeddings.unsqueeze(1),
            typology_embeddings.unsqueeze(0),
            dim=-1
        )
        typology_sim = typology_sim / self.temperature
        
        # Compute pairwise routing similarity
        routing_sim = F.cosine_similarity(
            routing_weights.unsqueeze(1),
            routing_weights.unsqueeze(0),
            dim=-1
        )
        
        # Encourage routing similarity to match typology similarity
        loss = F.mse_loss(routing_sim, typology_sim)
        
        return self.alpha * loss


# ============================================================================
# SPARSITY LOSS
# ============================================================================

class SparsityLoss(nn.Module):
    """
    Sparsity loss to encourage sparse routing (using fewer adapters).
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        target_sparsity: Optional[float] = None,
        loss_type: str = "l1"
    ):
        """
        Args:
            alpha: Weight for sparsity loss
            target_sparsity: Target fraction of active adapters (e.g., 0.2 for 20%)
            loss_type: Type of sparsity loss ("l1" or "target")
        """
        super().__init__()
        self.alpha = alpha
        self.target_sparsity = target_sparsity
        self.loss_type = loss_type
    
    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss.
        
        Args:
            routing_weights: Routing weights (batch_size, num_adapters)
        
        Returns:
            loss: Scalar sparsity loss
        """
        if self.loss_type == "l1":
            loss = routing_weights.abs().sum(dim=-1).mean()
        
        elif self.loss_type == "target":
            if self.target_sparsity is None:
                raise ValueError("target_sparsity must be specified for 'target' loss type")
            
            active_count = (routing_weights > 1e-5).sum(dim=-1).float()
            num_adapters = routing_weights.size(-1)
            target_count = torch.full_like(active_count, num_adapters * self.target_sparsity)
            loss = F.mse_loss(active_count, target_count)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.alpha * loss


# ============================================================================
# COMBINED TADR LOSS
# ============================================================================

class TADRLoss(nn.Module):
    """
    Combined loss function for TADR model.
    Combines task loss with optional regularization losses.
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        num_classes: Optional[int] = None,
        num_adapters: int = 10,
        use_load_balancing: bool = True,
        load_balancing_weight: float = 0.01,
        use_typology_reg: bool = False,
        typology_reg_weight: float = 0.01,
        use_sparsity: bool = False,
        sparsity_weight: float = 0.01,
        sparsity_target: Optional[float] = None
    ):
        """
        Args:
            task_type: Type of task
            num_classes: Number of classes
            num_adapters: Number of adapters
            use_load_balancing: Whether to use load balancing loss
            load_balancing_weight: Weight for load balancing loss
            use_typology_reg: Whether to use typology regularization
            typology_reg_weight: Weight for typology regularization
            use_sparsity: Whether to use sparsity loss
            sparsity_weight: Weight for sparsity loss
            sparsity_target: Target sparsity level
        """
        super().__init__()
        
        # Task loss
        self.task_loss = TaskLoss(task_type=task_type, num_classes=num_classes)
        
        # Load balancing loss
        self.use_load_balancing = use_load_balancing
        if use_load_balancing:
            self.load_balancing_loss = LoadBalancingLoss(
                num_adapters=num_adapters,
                alpha=load_balancing_weight,
                loss_type="mse"
            )
        
        # Typology regularization
        self.use_typology_reg = use_typology_reg
        if use_typology_reg:
            self.typology_reg_loss = TypologyRegularizationLoss(
                num_adapters=num_adapters,
                alpha=typology_reg_weight
            )
        
        # Sparsity loss
        self.use_sparsity = use_sparsity
        if use_sparsity:
            self.sparsity_loss = SparsityLoss(
                alpha=sparsity_weight,
                target_sparsity=sparsity_target,
                loss_type="target" if sparsity_target else "l1"
            )
        
        self.loss_weights = {
            'task': 1.0,
            'load_balancing': load_balancing_weight,
            'typology_reg': typology_reg_weight,
            'sparsity': sparsity_weight
        }
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        typology_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            routing_weights: Routing weights from all layers
            typology_embeddings: Typology embeddings
            attention_mask: Attention mask
            return_components: Whether to return individual loss components
        
        Returns:
            Dictionary with 'total' loss and optional components
        """
        # Task loss
        task_loss = self.task_loss(logits, labels, attention_mask)
        
        losses = {'task': task_loss}
        total_loss = task_loss
        
        # Load balancing loss
        if self.use_load_balancing and routing_weights is not None:
            lb_loss = self.load_balancing_loss(routing_weights)
            losses['load_balancing'] = lb_loss
            total_loss = total_loss + lb_loss
        
        # Typology regularization
        if self.use_typology_reg and routing_weights is not None and typology_embeddings is not None:
            if isinstance(routing_weights, list):
                last_routing = routing_weights[-1]
            else:
                last_routing = routing_weights
            
            typo_loss = self.typology_reg_loss(last_routing, typology_embeddings)
            losses['typology_reg'] = typo_loss
            total_loss = total_loss + typo_loss
        
        # Sparsity loss
        if self.use_sparsity and routing_weights is not None:
            if isinstance(routing_weights, list):
                sparsity_loss = sum(self.sparsity_loss(w) for w in routing_weights) / len(routing_weights)
            else:
                sparsity_loss = self.sparsity_loss(routing_weights)
            
            losses['sparsity'] = sparsity_loss
            total_loss = total_loss + sparsity_loss
        
        losses['total'] = total_loss
        
        if return_components:
            return losses
        else:
            return {'total': total_loss}
    
    def get_loss_summary(self, losses: Dict[str, torch.Tensor]) -> str:
        """Get formatted string of loss components."""
        summary = f"Total: {losses['total'].item():.4f}"
        if 'task' in losses:
            summary += f" | Task: {losses['task'].item():.4f}"
        if 'load_balancing' in losses:
            summary += f" | LB: {losses['load_balancing'].item():.4f}"
        if 'typology_reg' in losses:
            summary += f" | Typo: {losses['typology_reg'].item():.4f}"
        if 'sparsity' in losses:
            summary += f" | Sparse: {losses['sparsity'].item():.4f}"
        return summary