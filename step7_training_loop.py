"""
TADR Framework - Step 7: Training Loop
Complete training pipeline for TADR model with zero-shot evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import json
import os
from collections import defaultdict
import time
import pickle

# Import previous components
from step5_integration import CompleteTADRModel, create_tadr_model
from step6_loss_functions import TADRLoss

tqdm.disable = True
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# DATASET
# ============================================================================

class MultilingualDataset(Dataset):
    """Dataset for multilingual text classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        language_ids: List[str],
        tokenizer,
        max_length: int = 128
    ):
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


# ============================================================================
# TRAINER
# ============================================================================

class TADRTrainer:
    """Trainer class for TADR model."""
    
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
        accumulation_steps: int = 1,
        verbose: bool = True
    ):
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
        self.verbose = verbose
        
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        if scheduler is None and train_dataloader is not None:
            self.scheduler = self._create_scheduler(len(train_dataloader))
        else:
            self.scheduler = scheduler
        
        if loss_fn is None:
            self.loss_fn = TADRLoss(
                task_type="classification",
                num_classes=3,
                num_adapters=10,
                use_load_balancing=True,
                load_balancing_weight=0.01
            )
        else:
            self.loss_fn = loss_fn
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_acc_per_lang': []
        }
    
    def _create_optimizer(self , lr:float = 1e-4) -> torch.optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr = lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if self.verbose:
            num_params = sum(p.numel() for p in trainable_params)
            print(f"Optimizer: {num_params:,} trainable parameters , LR={lr}")
        return optimizer
    
    def _create_scheduler(self, steps_per_epoch: int, num_epochs: int = 10):
        num_training_steps = steps_per_epoch * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        if self.verbose:
            print(f"Scheduler: {num_training_steps} steps, {num_warmup_steps} warmup")
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        task_loss_sum = 0
        lb_loss_sum = 0
        num_batches = 0
        correct = 0
        total_samples = 0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.epoch + 1}/{self.epoch + 1}",
            disable=not self.verbose
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            language_ids = batch['language_ids']
            
            outputs = self.model(
                lang_ids=language_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_routing_info=True
            )
            
            routing_weights = None
            if 'routing_info' in outputs and outputs['routing_info']:
                routing_weights = [info['weights'] for info in outputs['routing_info']]
            
            loss_dict = self.loss_fn(
                logits=outputs['logits'],
                labels=labels,
                routing_weights=routing_weights,
                attention_mask=attention_mask,
                return_components=True
            )
            
            loss = loss_dict['total']
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            total_loss += loss.item() * self.accumulation_steps
            if 'task' in loss_dict:
                task_loss_sum += loss_dict['task'].item()
            if 'load_balancing' in loss_dict:
                lb_loss_sum += loss_dict['load_balancing'].item()
            num_batches += 1
            
            if self.verbose:
                progress_bar.set_postfix({
                    'loss': f"{total_loss / num_batches:.4f}",
                    'acc': f"{correct / total_samples:.3f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            if self.val_dataloader and self.global_step % self.eval_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()
                
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint('best_model.pt')
                    if self.verbose:
                        print(f"\n✅ New best accuracy: {self.best_val_acc:.4f}")
            
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': correct / total_samples,  # ← ADD THIS
            'task_loss': task_loss_sum / num_batches,
            'lb_loss': lb_loss_sum / num_batches if lb_loss_sum > 0 else 0
        }
        
        return metrics
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        if dataloader is None:
            dataloader = self.val_dataloader
        
        if dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        lang_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                language_ids = batch['language_ids']
                
                outputs = self.model(
                    lang_ids=language_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss_dict = self.loss_fn(
                    logits=outputs['logits'],
                    labels=labels,
                    return_components=False
                )
                
                total_loss += loss_dict['total'].item()
                
                preds = outputs['logits'].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                for i, lang_id in enumerate(language_ids):
                    lang_metrics[lang_id]['total'] += 1
                    if preds[i] == labels[i]:
                        lang_metrics[lang_id]['correct'] += 1
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        lang_acc = {
            lang: stats['correct'] / stats['total']
            for lang, stats in lang_metrics.items()
        }
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'language_accuracy': lang_acc
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Evaluation Results:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"\n  Per-language accuracy:")
            for lang, acc in sorted(lang_acc.items()):
                print(f"    {lang}: {acc:.4f}")
            print(f"{'='*70}\n")
        
        return metrics
    
    def train(self, num_epochs: int):
        """Main training loop."""
        if self.verbose:
            print(f"\n{'='*70}")
            print("Starting Training")
            print(f"{'='*70}")
            print(f"Epochs: {num_epochs}")
            print(f"Batches/epoch: {len(self.train_dataloader)}")
            if self.val_dataloader:
                print(f"Validation batches: {len(self.val_dataloader)}")
            print(f"Device: {self.device}")
            print(f"Output: {self.output_dir}")
            print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch()
            
            if self.verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Task Loss: {train_metrics['task_loss']:.4f}")
                if train_metrics['lb_loss'] > 0:
                    print(f"  LB Loss: {train_metrics['lb_loss']:.6f}")
            
            self.history['train_loss'].append(train_metrics['loss'])
            
            if self.val_dataloader:
                val_metrics = self.evaluate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_acc_per_lang'].append(val_metrics['language_accuracy'])
                
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
                    if self.verbose:
                        print(f"\n✅ Best model saved! Acc: {self.best_val_acc:.4f}")
            
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("Training Complete!")
            print(f"{'='*70}")
            print(f"Time: {total_time/60:.2f} minutes")
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Model saved to: {self.output_dir}")
            print(f"{'='*70}\n")
        
        self.save_training_history()
    
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
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, checkpoint_path)
        if self.verbose:
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
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if self.verbose:
            print(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc'],
            'val_acc_per_lang': self.history['val_acc_per_lang'],
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step
        }
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        history = convert_to_serializable(history)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if self.verbose:
            print(f"Training history saved: {history_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_preprocessed_data(data_dir: str = './data_zeroshot'):
    """Load preprocessed XNLI data."""
    
    def load_split(split_name):
        pkl_path = os.path.join(data_dir, f'xnli_{split_name}.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    train_data = load_split('train')
    val_data = load_split('validation')
    test_data = load_split('test')
    
    return train_data, val_data, test_data


def create_dataloaders(
    train_data: Dict,
    val_data: Dict,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
):
    """Create dataloaders from preprocessed data."""
    
    train_dataset = MultilingualDataset(
        texts=train_data['texts'],
        labels=train_data['labels'],
        language_ids=train_data['lang_ids'],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = MultilingualDataset(
        texts=val_data['texts'],
        labels=val_data['labels'],
        language_ids=val_data['lang_ids'],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    print(f"Dataloaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# ============================================================================
# MAIN - THIS IS WHAT YOU ASKED FOR!
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("TADR Framework - Step 7: Training")
    print("🚀"*35 + "\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Paths
    DATA_DIR = './data_zeroshot'
    OUTPUT_DIR = './tadr_checkpoints'
    WALS_FILE = 'wals_features.csv'
    
    # Model config
    MODEL_NAME = 'xlm-roberta-base'
    NUM_ADAPTERS = 8
    ADAPTER_BOTTLENECK = 48
    NUM_CLASSES = 3 
    GATING_TYPE = 'softmax'
    NUM_ADAPTER_LAYERS = 4 
    UNFREEZING_STRATEGY = 'full'  # Options: minimal, efficient, aggressive, full
    
    # Training config
    BATCH_SIZE = 8
    MAX_LENGTH = 128
    EVAL_STEPS = 200
    SAVE_STEPS = 500
    ACCUMULATION_STEPS = 2
    HYPERPARAMS = {
        'minimal': {
            'lr': 5e-4,          # High LR for small parameter set
            'epochs': 5,         # More epochs needed
            'classifier_lr_multiplier': 10
        },
        'efficient': {
            'lr': 5e-5,          # Standard BERT fine-tuning LR
            'epochs': 3,         # Moderate epochs
            'classifier_lr_multiplier': 5
        },
        'aggressive': {
            'lr': 3e-5,          # Lower LR for more parameters
            'epochs': 3,         # Fewer epochs (learns faster)
            'classifier_lr_multiplier': 3
        },
        'full': {
            'lr': 2e-5,          # Lowest LR for all parameters
            'epochs': 3,         # Fewest epochs (learns very fast)
            'classifier_lr_multiplier': 1
        }
    }
    LEARNING_RATE = HYPERPARAMS[UNFREEZING_STRATEGY]['lr']
    NUM_EPOCHS = HYPERPARAMS[UNFREEZING_STRATEGY]['epochs']
    CLASSIFIER_LR_MULTIPLIER = HYPERPARAMS[UNFREEZING_STRATEGY]['classifier_lr_multiplier']

    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Num adapters: {NUM_ADAPTERS}")
    print(f"  Adapter layers: {NUM_ADAPTER_LAYERS}")
    print(f"  Unfreezing strategy: {UNFREEZING_STRATEGY}") 
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Step 1: Loading Data")
    print(f"{'='*70}\n")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory '{DATA_DIR}' not found!")
        print(f"Please run 'python data_preparation_xnli.py' first.")
        exit(1)
    
    train_data, val_data, test_data = load_preprocessed_data(DATA_DIR)
    
    print(f"✅ Data loaded:")
    print(f"  Train: {len(train_data['texts'])} samples")
    print(f"  Val: {len(val_data['texts'])} samples")
    print(f"  Test: {len(test_data['texts'])} samples")
    
    # ========================================================================
    # STEP 2: CREATE MODEL
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Step 2: Creating TADR Model")
    print(f"{'='*70}\n")
    
    if not os.path.exists(WALS_FILE):
        print(f"❌ Error: WALS features file '{WALS_FILE}' not found!")
        print(f"Please ensure wals_features.csv exists in the current directory.")
        exit(1)
    
    model = create_tadr_model(
        model_name=MODEL_NAME,
        feature_file=WALS_FILE,
        num_adapters=NUM_ADAPTERS,
        adapter_bottleneck=ADAPTER_BOTTLENECK,
        num_classes=NUM_CLASSES,
        gating_type=GATING_TYPE,
        num_adapter_layers=NUM_ADAPTER_LAYERS,
        device=DEVICE,
        unfreezing_strategy=UNFREEZING_STRATEGY
    )
    
    print(f"✅ Model created successfully!")
    
    # ========================================================================
    # STEP 3: CREATE DATALOADERS
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Step 3: Creating DataLoaders")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_loader, val_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    # ========================================================================
    # STEP 4: CREATE LOSS FUNCTION
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Step 4: Creating Loss Function")
    print(f"{'='*70}\n")
    
    loss_fn = TADRLoss(
        task_type="classification",
        num_classes=NUM_CLASSES,
        num_adapters=NUM_ADAPTERS,
        use_load_balancing=False,
        load_balancing_weight = 0.0
    )
    
    print("✅ Loss function created")
    
    # ========================================================================
    # STEP 5: CREATE TRAINER
    # ========================================================================

    print(f"\n{'='*70}")
    print("Step 5: Creating Optimizer and Scheduler")
    print(f"{'='*70}\n")

    # Manual optimizer with CORRECT learning rate
    # Create optimizer with separate learning rates
    classifier_params = []
    adapter_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                adapter_params.append(param)
    
    optimizer = AdamW([
        {'params': adapter_params, 'lr': LEARNING_RATE},
        {'params': classifier_params, 'lr': LEARNING_RATE * CLASSIFIER_LR_MULTIPLIER}  # 10x higher for classifier
    ], weight_decay=0.01)
    
    print(f"Optimizer: base_lr={LEARNING_RATE}, classifier_lr={LEARNING_RATE*CLASSIFIER_LR_MULTIPLIER}")

    # Manual scheduler with CORRECT num_epochs
    num_training_steps = len(train_loader) * NUM_EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)

    print(f"\n🔧 Scheduler Debug:")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  len(train_loader): {len(train_loader)}")
    print(f"  num_training_steps: {num_training_steps}")
    print(f"  Expected: {len(train_loader) * NUM_EPOCHS}")
    print(f"  Match: {num_training_steps == len(train_loader) * NUM_EPOCHS}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"✅ Optimizer created:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")

    
    print(f"\n{'='*70}")
    print("Step 6: Creating Trainer")
    print(f"{'='*70}\n")
    
    trainer = TADRTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        logging_steps=100,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        accumulation_steps=ACCUMULATION_STEPS,
        verbose=True
    )
    
    print("✅ Trainer created")
    
    # ========================================================================
    # STEP 6: TRAIN!
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Step 7: Training")
    print(f"{'='*70}\n")
    
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nTraining Summary:")
    print(f"  Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Total epochs: {NUM_EPOCHS}")
    print(f"  Total training steps: {trainer.global_step}")

    print(f"\n💾 Model saved to:")
    print(f"  {OUTPUT_DIR}/best_model.pt (best validation accuracy)")
    print(f"  {OUTPUT_DIR}/checkpoint_epoch_*.pt (per-epoch checkpoints)")
    print(f"  {OUTPUT_DIR}/training_history.json (training metrics)")

    print(f"\n{'='*70}\n")