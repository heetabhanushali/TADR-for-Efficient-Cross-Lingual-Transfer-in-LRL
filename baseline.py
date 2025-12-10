import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

def train_baseline_with_monitoring():
    """
    Baseline with REAL-TIME accuracy monitoring
    """
    print("="*70)
    print("🎯 BASELINE: XLM-RoBERTa (with live accuracy tracking)")
    print("="*70)

    # Set seed
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n📂 Loading data...")

    import pickle
    with open('./data_zeroshot/xnli_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data_zeroshot/xnli_validation.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('./data_zeroshot/xnli_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print(f"Train: {len(train_data['texts'])} samples")
    print(f"Test: {len(test_data['texts'])} samples")

    # Check label balance
    from collections import Counter
    train_labels = Counter(train_data['labels'])
    print(f"Label distribution: {train_labels}")

    # ============================================================
    # 2. MODEL
    # ============================================================
    print("\n🔨 Creating model...")

    from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=3
    )
    model = model.to(device)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    # ============================================================
    # 3. DATASET
    # ============================================================
    from torch.utils.data import Dataset, DataLoader

    class SimpleDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    train_dataset = SimpleDataset(train_data['texts'], train_data['labels'], tokenizer)
    val_dataset = SimpleDataset(val_data['texts'], val_data['labels'], tokenizer)
    test_dataset = SimpleDataset(test_data['texts'], test_data['labels'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ============================================================
    # 4. OPTIMIZER (FIXED IMPORT!)
    # ============================================================
    print("\n⚙️ Setting up optimizer...")

    # Use torch.optim.AdamW instead of transformers.AdamW
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"LR: 2e-5 | Epochs: {num_epochs} | Steps/epoch: {len(train_loader)}")

    # ============================================================
    # 5. TRAINING WITH LIVE MONITORING
    # ============================================================
    print("\n🚀 Training (you can stop anytime with Runtime > Interrupt)")
    print("Watch the accuracy - it should INCREASE each epoch!")
    print("If stuck at 33.3%, stop and we'll debug\n")

    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print('='*70)

        # ============================================================
        # TRAINING PHASE
        # ============================================================
        model.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        running_loss = 0
        running_correct = 0
        running_total = 0

        print_every = 50  # Print every 50 batches

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Calculate batch accuracy
            batch_correct = (predictions == labels).sum().item()
            batch_total = labels.size(0)
            batch_acc = batch_correct / batch_total

            # Accumulate
            epoch_correct += batch_correct
            epoch_total += batch_total
            epoch_loss += loss.item()

            running_correct += batch_correct
            running_total += batch_total
            running_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # PRINT PROGRESS EVERY N BATCHES
            if (batch_idx + 1) % print_every == 0:
                running_acc = running_correct / running_total
                avg_running_loss = running_loss / print_every

                print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                      f"Loss: {avg_running_loss:.4f} | "
                      f"Acc: {running_acc:.4f} ({running_acc*100:.1f}%) | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

                # Reset running metrics
                running_loss = 0
                running_correct = 0
                running_total = 0

        # EPOCH SUMMARY
        epoch_avg_loss = epoch_loss / len(train_loader)
        epoch_avg_acc = epoch_correct / epoch_total

        print(f"\n{'─'*70}")
        print(f"📊 EPOCH {epoch+1} TRAINING SUMMARY:")
        print(f"   Loss: {epoch_avg_loss:.4f}")
        print(f"   Accuracy: {epoch_avg_acc:.4f} ({epoch_avg_acc*100:.1f}%)")
        print(f"{'─'*70}")

        # ============================================================
        # VALIDATION PHASE
        # ============================================================
        print("\n🔍 Validating...")
        model.eval()

        val_correct = 0
        val_total = 0
        val_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                val_predictions.extend(predictions.cpu().numpy())
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_dist = Counter(val_predictions)

        print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"   Prediction distribution: {val_dist}")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   ✅ NEW BEST: {best_val_acc:.4f}")
        else:
            print(f"   📌 Best so far: {best_val_acc:.4f}")

        # WARNING if stuck
        if val_acc < 0.40:
            print(f"   ⚠️  WARNING: Accuracy is low. Model might not be learning well.")

        print()

    # ============================================================
    # 6. FINAL TEST EVALUATION
    # ============================================================
    print("\n" + "="*70)
    print("📊 FINAL TEST EVALUATION")
    print("="*70)

    model.eval()

    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    test_lang_ids = test_data['lang_ids']
    lang_correct = {}
    lang_total = {}

    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

            # Per-language
            batch_size = len(predictions)
            start_idx = batch_idx * 32
            end_idx = start_idx + batch_size
            batch_langs = test_lang_ids[start_idx:end_idx]

            for i, lang in enumerate(batch_langs):
                if lang not in lang_correct:
                    lang_correct[lang] = 0
                    lang_total[lang] = 0

                lang_total[lang] += 1
                if predictions[i] == labels[i]:
                    lang_correct[lang] += 1

            batch_idx += 1

    # Results
    test_acc = test_correct / test_total
    pred_dist = Counter(all_predictions)
    label_dist = Counter(all_labels)

    print(f"\n✅ OVERALL TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nPrediction distribution: {pred_dist}")
    print(f"Label distribution:      {label_dist}")

    print(f"\n📊 PER-LANGUAGE RESULTS:")
    print(f"{'Language':<10} {'Accuracy':<10} {'Samples':<10}")
    print("─"*30)

    lang_accs = {}
    for lang in sorted(lang_total.keys()):
        acc = lang_correct[lang] / lang_total[lang]
        lang_accs[lang] = acc
        print(f"{lang:<10} {acc:.4f} ({acc*100:.1f}%)  {lang_total[lang]:<10}")

    avg_acc = np.mean(list(lang_accs.values()))
    print("─"*30)
    print(f"{'AVERAGE':<10} {avg_acc:.4f} ({avg_acc*100:.1f}%)")

    # ============================================================
    # 7. SAVE RESULTS
    # ============================================================
    results = {
        'model': 'XLM-RoBERTa Baseline',
        'overall_accuracy': test_acc,
        'average_accuracy': avg_acc,
        'language_accuracies': lang_accs,
        'best_val_accuracy': best_val_acc,
        'prediction_distribution': dict(pred_dist),
        'label_distribution': dict(label_dist),
        'timestamp': datetime.now().isoformat()
    }

    save_path = '/content/drive/MyDrive/TADR_Results/baseline_results_final.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {save_path}")

    # ============================================================
    # 8. QUICK COMPARISON PREVIEW
    # ============================================================
    print("\n" + "="*70)
    print("📝 QUICK COMPARISON (for your paper)")
    print("="*70)
    print(f"\nBaseline (XLM-R): {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Your TADR:        0.6647 (66.5%)")
    print(f"Improvement:      {0.6647 - test_acc:.4f} ({(0.6647 - test_acc)*100:.1f}%)")
    print("="*70)

    return results

# ============================================================
# RUN THE BASELINE
# ============================================================
print("\n🎬 Starting baseline training with live monitoring")
print("💡 TIP: Watch the accuracy numbers!")
print("   - Should start around 33-40% in first epoch")
print("   - Should increase to 50-60% by end of training")
print("   - If stuck at 33%, stop immediately (Ctrl+C)\n")

baseline_results = train_baseline_with_monitoring()