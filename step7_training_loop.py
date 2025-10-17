"""
TADR Framework - Step 7: Training Loop
Complete training pipeline for TADR model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import json
import os
from collections import defaultdict
import time

# Import previous components
# from step1_typology_module import TypologyFeatureModule
# from step5_integration import TADRModel
# from step6_loss_functions import TADRLoss


class MultilingualDataset(Dataset):
    """
    Dataset for multilingual text classification/sequence labeling.
    Handles multiple languages with language ID tracking.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        language_ids: List[str],
        tokenizer,
        max_length: int = 128
    ):
        """
        Args:
            texts: List of text samples
            labels: List of labels
            language_ids: List of language IDs (e.g., ['en', 'hi', 'zh'])
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.language_ids = language_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts) == len(labels) == len(language_ids), \
            "Texts, labels, and language_ids must have same length"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        lang_id = self.language_ids[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'language_id': lang_id
        }


def collate_fn(batch):
    """Custom collate function to handle language IDs."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'language_ids': [item['language_id'] for item in batch]
    }


class TADRTrainer:
    """
    Trainer class for TADR model.
    Handles training loop, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cuda',
        output_dir: str = './checkpoints',
        logging_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 1
    ):
        """
        Args:
            model: TADR model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Device to use
            output_dir: Directory for checkpoints
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            max_grad_norm: Gradient clipping threshold
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        
        # Setup optimizer if not provided
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Setup scheduler if not provided
        if scheduler is None and train_dataloader is not None:
            self.scheduler = self._create_scheduler(len(train_dataloader))
        else:
            self.scheduler = scheduler
        
        # Setup loss function if not provided
        if loss_fn is None:
            from step6_loss_functions import TADRLoss
            self.loss_fn = TADRLoss(
                task_type="classification",
                num_classes=3,  # Default
                num_adapters=10,
                use_load_balancing=True,
                load_balancing_weight=0.01
            )
        else:
            self.loss_fn = loss_fn
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.routing_stats = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for trainable parameters only."""
        # Only optimize adapter and router parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        print(f"Optimizer created with {len(trainable_params)} parameter groups")
        return optimizer
    
    def _create_scheduler(self, steps_per_epoch: int, num_epochs: int = 10):
        """Create learning rate scheduler."""
        num_training_steps = steps_per_epoch * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"Scheduler created: {num_training_steps} steps, {num_warmup_steps} warmup")
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        task_loss_sum = 0
        lb_loss_sum = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            language_ids = batch['language_ids']
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_ids=language_ids,
                return_routing_info=True
            )
            
            # Get routing weights from all layers
            routing_weights = [info['weights'] for info in outputs.get('routing_info', [])]
            
            # Compute loss
            loss_dict = self.loss_fn(
                logits=outputs['logits'],
                labels=labels,
                routing_weights=routing_weights,
                typology_embeddings=None,  # Already embedded
                attention_mask=attention_mask,
                return_components=True
            )
            
            loss = loss_dict['total']
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Logging
            total_loss += loss.item() * self.accumulation_steps
            if 'task' in loss_dict:
                task_loss_sum += loss_dict['task'].item()
            if 'load_balancing' in loss_dict:
                lb_loss_sum += loss_dict['load_balancing'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Periodic logging
            if self.global_step % self.logging_steps == 0:
                self._log_training_step(loss_dict)
            
            # Periodic evaluation
            if self.val_dataloader and self.global_step % self.eval_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()  # Back to training mode
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # Periodic saving
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        # Epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'task_loss': task_loss_sum / num_batches,
            'lb_loss': lb_loss_sum / num_batches if lb_loss_sum > 0 else 0
        }
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        # Language-specific metrics
        lang_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                language_ids = batch['language_ids']
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    language_ids=language_ids
                )
                
                # Compute loss
                loss_dict = self.loss_fn(
                    logits=outputs['logits'],
                    labels=labels,
                    return_components=False
                )
                
                total_loss += loss_dict['total'].item()
                
                # Predictions
                preds = outputs['logits'].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Per-language metrics
                for i, lang_id in enumerate(language_ids):
                    lang_metrics[lang_id]['total'] += 1
                    if preds[i] == labels[i]:
                        lang_metrics[lang_id]['correct'] += 1
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        
        # Per-language accuracy
        lang_acc = {
            lang: stats['correct'] / stats['total']
            for lang, stats in lang_metrics.items()
        }
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'language_accuracy': lang_acc
        }
        
        print(f"\nValidation Metrics:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Per-language accuracy:")
        for lang, acc in lang_acc.items():
            print(f"    {lang}: {acc:.4f}")
        
        return metrics
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches per epoch: {len(self.train_dataloader)}")
        if self.val_dataloader:
            print(f"Validation batches: {len(self.val_dataloader)}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            print(f"\nTraining Metrics (Epoch {epoch + 1}):")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Task Loss: {train_metrics['task_loss']:.4f}")
            if train_metrics['lb_loss'] > 0:
                print(f"  Load Balancing Loss: {train_metrics['lb_loss']:.6f}")
            
            # Evaluate
            if self.val_dataloader:
                val_metrics = self.evaluate()
                self.val_losses.append(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
                    print(f"\n✅ New best model saved! (val_loss: {val_metrics['loss']:.4f})")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            self.train_losses.append(train_metrics['loss'])
        
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to: {self.output_dir}")
        print("="*70 + "\n")
        
        # Save training history
        self.save_training_history()
    
    def _log_training_step(self, loss_dict: Dict[str, torch.Tensor]):
        """Log training step information."""
        log_str = f"Step {self.global_step} | "
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                log_str += f"{name}: {value.item():.4f} | "
        
        # Get current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        log_str += f"lr: {lr:.2e}"
        
        print(log_str)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step
        }
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {history_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    train_lang_ids: List[str],
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    val_lang_ids: Optional[List[str]] = None,
    tokenizer = None,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        train_lang_ids: Training language IDs
        val_texts: Validation texts
        val_labels: Validation labels
        val_lang_ids: Validation language IDs
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
    
    Returns:
        train_dataloader, val_dataloader (or None)
    """
    # Training dataset
    train_dataset = MultilingualDataset(
        texts=train_texts,
        labels=train_labels,
        language_ids=train_lang_ids,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    # Validation dataset
    val_dataloader = None
    if val_texts is not None:
        val_dataset = MultilingualDataset(
            texts=val_texts,
            labels=val_labels,
            language_ids=val_lang_ids,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    
    print(f"Created dataloaders:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Training batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader


def setup_training(
    model,
    train_dataloader,
    val_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    device: str = 'cuda',
    output_dir: str = './checkpoints',
    **trainer_kwargs
) -> TADRTrainer:
    """
    Setup trainer with optimizer and scheduler.
    
    Args:
        model: TADR model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        device: Device
        output_dir: Output directory
        **trainer_kwargs: Additional trainer arguments
    
    Returns:
        Configured TADRTrainer
    """
    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create trainer
    trainer = TADRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        **trainer_kwargs
    )
    
    return trainer


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def test_dataset():
    """Test multilingual dataset."""
    print("="*60)
    print("Testing Multilingual Dataset")
    print("="*60)
    
    from transformers import AutoTokenizer
    
    # Sample data
    texts = [
        "This is a positive review.",
        "यह एक अच्छी समीक्षा है।",
        "这是一个积极的评论。",
        "This is negative.",
        "यह नकारात्मक है।"
    ]
    labels = [1, 1, 1, 0, 0]
    lang_ids = ['en', 'hi', 'zh', 'en', 'hi']
    
    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    dataset = MultilingualDataset(
        texts=texts,
        labels=labels,
        language_ids=lang_ids,
        tokenizer=tokenizer,
        max_length=64
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test batch
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Label: {sample['labels']}")
    print(f"  Language ID: {sample['language_id']}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Language IDs: {batch['language_ids']}")
    
    print("\n✅ Dataset test passed!")


def demo_training_setup():
    """Demonstrate training setup."""
    print("\n" + "="*60)
    print("Training Setup Demo")
    print("="*60)
    
    print("""
# 1. Prepare data
train_texts = ["text1", "text2", ...]
train_labels = [0, 1, ...]
train_lang_ids = ["en", "hi", ...]

val_texts = ["val_text1", ...]
val_labels = [1, ...]
val_lang_ids = ["zh", ...]

# 2. Create dataloaders
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

train_loader, val_loader = create_dataloaders(
    train_texts=train_texts,
    train_labels=train_labels,
    train_lang_ids=train_lang_ids,
    val_texts=val_texts,
    val_labels=val_labels,
    val_lang_ids=val_lang_ids,
    tokenizer=tokenizer,
    batch_size=16,
    max_length=128
)

# 3. Setup trainer
trainer = setup_training(
    model=tadr_model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    learning_rate=1e-4,
    num_epochs=10,
    device='cuda',
    output_dir='./checkpoints',
    logging_steps=100,
    eval_steps=500,
    save_steps=1000
)

# 4. Train
trainer.train(num_epochs=10)

# 5. Load best model
trainer.load_checkpoint('checkpoints/best_model.pt')

# 6. Evaluate
val_metrics = trainer.evaluate()
    """)
    
    print("\n✅ Training setup demo complete!")


def demo_full_pipeline():
    """Demonstrate complete training pipeline."""
    print("\n" + "="*60)
    print("Complete Training Pipeline")
    print("="*60)
    
    print("""
# Complete end-to-end example:

from step1_typology_module import TypologicalFeatureLoader, TypologyFeatureModule
from step5_integration import create_tadr_model
from step6_loss_functions import TADRLoss
from step7_training_loop import create_dataloaders, setup_training

# 1. Create TADR model
model = create_tadr_model(
    base_model_name="xlm-roberta-base",
    typology_feature_file="typology_features.csv",
    num_adapters=10,
    adapter_bottleneck=64,
    num_classes=3,
    gating_type="topk"
)

# 2. Prepare loss function
loss_fn = TADRLoss(
    task_type="classification",
    num_classes=3,
    num_adapters=10,
    use_load_balancing=True,
    load_balancing_weight=0.01,
    use_typology_reg=True,
    typology_reg_weight=0.005
)

# 3. Load and prepare data
# (Assuming you have train/val data loaded)
train_loader, val_loader = create_dataloaders(
    train_texts, train_labels, train_lang_ids,
    val_texts, val_labels, val_lang_ids,
    tokenizer=model.base_model.tokenizer,
    batch_size=16
)

# 4. Setup trainer
trainer = setup_training(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    learning_rate=1e-4,
    num_epochs=10,
    device='cuda'
)

trainer.loss_fn = loss_fn  # Use custom loss

# 5. Train!
trainer.train(num_epochs=10)

# 6. Test zero-shot transfer
test_loader = create_test_loader(
    test_texts,
    test_labels,
    ["sw"] * len(test_texts),  # Swahili - unseen language!
    tokenizer
)

trainer.model.eval()
# Model will route based on Swahili's typology automatically!
    """)
    
    print("\n✅ Full pipeline demo complete!")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("TADR Framework - Step 7: Training Loop")
    print("🚀" * 30)
    
    # Run tests
    test_dataset()
    demo_training_setup()
    demo_full_pipeline()
    
    print("\n" + "="*60)
    print("✅ Step 7 Complete!")
    print("="*60)
    print("\nPhase 3 (Training Pipeline) Complete!")
    print("\nAll components:")
    print("  ✅ Step 6: Loss Functions")
    print("  ✅ Step 7: Training Loop")
    print("\nNext steps:")
    print("  → Step 8: Zero-Shot Transfer Testing")
    print("  → Step 9: Analysis & Visualization")
    print("="*60 + "\n")