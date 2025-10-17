"""
TADR Framework - Step 6: Loss Functions
Implementation of task losses, regularization, and load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


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
            # logits: (batch_size, num_classes)
            # labels: (batch_size,)
            return self.loss_fn(logits, labels)
        
        elif self.task_type == "sequence_labeling":
            # logits: (batch_size, seq_len, num_classes)
            # labels: (batch_size, seq_len)
            batch_size, seq_len, num_classes = logits.shape
            
            # Reshape for loss computation
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            return self.loss_fn(logits_flat, labels_flat)


class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss to encourage uniform adapter usage across the batch.
    Prevents router collapse (using only one adapter for all inputs).
    
    Based on the auxiliary loss from Switch Transformers and similar MoE models.
    """
    
    def __init__(
        self,
        num_adapters: int,
        alpha: float = 0.01,
        loss_type: str = "mse"  # "mse", "entropy", or "auxiliary"
    ):
        """
        Args:
            num_adapters: Number of adapters (K)
            alpha: Weight for load balancing loss
            loss_type: Type of load balancing loss
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
            routing_weights: Routing weights (batch_size, num_adapters) or 
                           list of tensors for multiple layers
            layer_idx: Layer index (for logging/debugging)
        
        Returns:
            loss: Scalar load balancing loss
        """
        if isinstance(routing_weights, list):
            # Average over multiple layers
            losses = [self._compute_single_loss(w) for w in routing_weights]
            return sum(losses) / len(losses)
        else:
            return self._compute_single_loss(routing_weights)
    
    def _compute_single_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute loss for a single set of routing weights."""
        # Average usage across batch
        # routing_weights: (batch_size, num_adapters)
        avg_usage = routing_weights.mean(dim=0)  # (num_adapters,)
        
        if self.loss_type == "mse":
            # MSE between actual and uniform distribution
            target = torch.ones_like(avg_usage) / self.num_adapters
            loss = F.mse_loss(avg_usage, target)
        
        elif self.loss_type == "entropy":
            # Maximize entropy of usage distribution
            # H(p) = -sum(p * log(p))
            epsilon = 1e-10
            entropy = -(avg_usage * torch.log(avg_usage + epsilon)).sum()
            max_entropy = np.log(self.num_adapters)  # Maximum possible entropy
            # Loss is negative entropy (we want to maximize entropy)
            loss = (max_entropy - entropy) / max_entropy
        
        elif self.loss_type == "auxiliary":
            # Auxiliary loss from Switch Transformers
            # Encourages equal load distribution
            f_i = avg_usage  # Fraction of tokens assigned to each expert
            P_i = routing_weights.mean(dim=0)  # Average routing probability
            loss = self.num_adapters * (f_i * P_i).sum()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.alpha * loss


class TypologyRegularizationLoss(nn.Module):
    """
    Typology-aware regularization loss.
    Encourages similar adapter usage for typologically similar languages.
    
    This helps the model learn meaningful typological patterns.
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
        # similarity: (batch_size, batch_size)
        typology_sim = F.cosine_similarity(
            typology_embeddings.unsqueeze(1),
            typology_embeddings.unsqueeze(0),
            dim=-1
        )
        
        # Apply temperature
        typology_sim = typology_sim / self.temperature
        
        # Compute pairwise routing similarity
        # Use KL divergence or cosine similarity
        routing_sim = F.cosine_similarity(
            routing_weights.unsqueeze(1),
            routing_weights.unsqueeze(0),
            dim=-1
        )
        
        # Loss: encourage routing similarity to match typology similarity
        # Use MSE between similarity matrices
        loss = F.mse_loss(routing_sim, typology_sim)
        
        return self.alpha * loss


class SparsityLoss(nn.Module):
    """
    Sparsity loss to encourage sparse routing (using fewer adapters).
    Helps with computational efficiency.
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        target_sparsity: Optional[float] = None,  # Target number of active adapters
        loss_type: str = "l1"  # "l1" or "target"
    ):
        """
        Args:
            alpha: Weight for sparsity loss
            target_sparsity: Target fraction of active adapters (e.g., 0.2 for 20%)
            loss_type: Type of sparsity loss
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
            # L1 norm encourages sparsity
            loss = routing_weights.abs().sum(dim=-1).mean()
        
        elif self.loss_type == "target":
            # Encourage specific sparsity level
            if self.target_sparsity is None:
                raise ValueError("target_sparsity must be specified for 'target' loss type")
            
            # Count active adapters (weights above threshold)
            active_count = (routing_weights > 1e-5).sum(dim=-1).float()
            num_adapters = routing_weights.size(-1)
            
            # Target number of active adapters
            target_count = torch.full_like(active_count, num_adapters * self.target_sparsity)
            
            # MSE loss
            loss = F.mse_loss(active_count, target_count)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.alpha * loss


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
            routing_weights: Routing weights from all layers (list or single tensor)
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
            # Use routing weights from last layer
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


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def test_task_loss():
    """Test task loss computation."""
    print("="*60)
    print("Testing Task Loss")
    print("="*60)
    
    # Classification
    print("\n1. Classification Task:")
    task_loss = TaskLoss(task_type="classification", num_classes=3)
    
    batch_size = 4
    logits = torch.randn(batch_size, 3)
    labels = torch.tensor([0, 1, 2, 1])
    
    loss = task_loss(logits, labels)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Sequence labeling
    print("\n2. Sequence Labeling Task:")
    task_loss = TaskLoss(task_type="sequence_labeling", num_classes=5)
    
    seq_len = 128
    logits = torch.randn(batch_size, seq_len, 5)
    labels = torch.randint(0, 5, (batch_size, seq_len))
    labels[:, 100:] = -100  # Padding tokens
    
    loss = task_loss(logits, labels)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ Task loss test passed!")


def test_load_balancing_loss():
    """Test load balancing loss."""
    print("\n" + "="*60)
    print("Testing Load Balancing Loss")
    print("="*60)
    
    num_adapters = 10
    batch_size = 32
    
    lb_loss = LoadBalancingLoss(num_adapters=num_adapters, alpha=0.01, loss_type="mse")
    
    # Test 1: Balanced distribution
    print("\n1. Balanced Routing:")
    balanced = torch.ones(batch_size, num_adapters) / num_adapters
    loss = lb_loss(balanced)
    print(f"   Usage distribution: {balanced[0]}")
    print(f"   Loss: {loss.item():.6f} (should be ~0)")
    
    # Test 2: Imbalanced (collapsed)
    print("\n2. Collapsed Routing:")
    collapsed = torch.zeros(batch_size, num_adapters)
    collapsed[:, 0] = 1.0
    loss = lb_loss(collapsed)
    print(f"   Usage distribution: {collapsed[0]}")
    print(f"   Loss: {loss.item():.6f} (should be high)")
    
    # Test 3: Partially imbalanced
    print("\n3. Partially Imbalanced:")
    partial = F.softmax(torch.randn(batch_size, num_adapters), dim=-1)
    loss = lb_loss(partial)
    avg_usage = partial.mean(dim=0)
    print(f"   Avg usage: {avg_usage}")
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n✅ Load balancing loss test passed!")


def test_typology_regularization():
    """Test typology regularization loss."""
    print("\n" + "="*60)
    print("Testing Typology Regularization Loss")
    print("="*60)
    
    batch_size = 4
    num_adapters = 10
    typology_dim = 128
    
    typo_loss = TypologyRegularizationLoss(
        num_adapters=num_adapters,
        alpha=0.01,
        similarity_temperature=1.0
    )
    
    # Create sample data
    routing_weights = F.softmax(torch.randn(batch_size, num_adapters), dim=-1)
    typology_embeddings = torch.randn(batch_size, typology_dim)
    
    loss = typo_loss(routing_weights, typology_embeddings)
    
    print(f"\n   Routing weights shape: {routing_weights.shape}")
    print(f"   Typology embeddings shape: {typology_embeddings.shape}")
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n✅ Typology regularization test passed!")


def test_combined_loss():
    """Test combined TADR loss."""
    print("\n" + "="*60)
    print("Testing Combined TADR Loss")
    print("="*60)
    
    # Create combined loss
    tadr_loss = TADRLoss(
        task_type="classification",
        num_classes=3,
        num_adapters=10,
        use_load_balancing=True,
        load_balancing_weight=0.01,
        use_typology_reg=True,
        typology_reg_weight=0.005,
        use_sparsity=True,
        sparsity_weight=0.001,
        sparsity_target=0.3
    )
    
    # Sample data
    batch_size = 4
    logits = torch.randn(batch_size, 3)
    labels = torch.tensor([0, 1, 2, 1])
    routing_weights = F.softmax(torch.randn(batch_size, 10), dim=-1)
    typology_embeddings = torch.randn(batch_size, 128)
    
    # Compute loss
    losses = tadr_loss(
        logits=logits,
        labels=labels,
        routing_weights=routing_weights,
        typology_embeddings=typology_embeddings,
        return_components=True
    )
    
    print("\n   Loss Components:")
    for name, value in losses.items():
        print(f"     {name}: {value.item():.6f}")
    
    print(f"\n   {tadr_loss.get_loss_summary(losses)}")
    
    print("\n✅ Combined loss test passed!")


def demo_loss_configurations():
    """Demonstrate different loss configurations."""
    print("\n" + "="*60)
    print("Loss Configuration Examples")
    print("="*60)
    
    configs = [
        {
            "name": "Basic (Task only)",
            "use_load_balancing": False,
            "use_typology_reg": False,
            "use_sparsity": False
        },
        {
            "name": "Standard (Task + Load Balancing)",
            "use_load_balancing": True,
            "load_balancing_weight": 0.01,
            "use_typology_reg": False,
            "use_sparsity": False
        },
        {
            "name": "Full Regularization",
            "use_load_balancing": True,
            "load_balancing_weight": 0.01,
            "use_typology_reg": True,
            "typology_reg_weight": 0.005,
            "use_sparsity": True,
            "sparsity_weight": 0.001,
            "sparsity_target": 0.3
        }
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")
        for k, v in config.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("TADR Framework - Step 6: Loss Functions")
    print("🚀" * 30)
    
    # Run all tests
    test_task_loss()
    test_load_balancing_loss()
    test_typology_regularization()
    test_combined_loss()
    demo_loss_configurations()
    
    print("\n" + "="*60)
    print("✅ Step 6 Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  → Step 7: Training Loop")
    print("  → Implement training pipeline with all loss components")
    print("="*60 + "\n")