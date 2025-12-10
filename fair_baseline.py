import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from collections import Counter

def train_fair_baseline():
    """
    FAIR BASELINE: XLM-R with only last 2 layers trainable
    ~3M parameters (matches TADR's 3.8M)
    """
    print("="*70)
    print("🎯 FAIR BASELINE: XLM-R (Last 2 Layers Only)")
    print("="*70)

    # Set seed
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

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
    print(f"Val: {len(val_data['texts'])} samples")
    print(f"Test: {len(test_data['texts'])} samples")

    # ============================================================
    # 2. MODEL WITH SELECTIVE FREEZING
    # ============================================================
    print("\n🔨 Creating model with selective freezing...")

    from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=3
    )

    # FREEZE EVERYTHING FIRST
    for param in model.parameters():
        param.requires_grad = False

    print("\n🔒 Freezing strategy:")
    print("  Layers 0-9: FROZEN ❄️")
    print("  Layers 10-11: TRAINABLE 🔥")
    print("  Classifier: TRAINABLE 🔥")

    # UNFREEZE ONLY LAST 2 TRANSFORMER LAYERS (10 and 11)
    for layer_idx in [10, 11]:
        for param in model.roberta.encoder.layer[layer_idx].parameters():
            param.requires_grad = True

    # UNFREEZE CLASSIFIER
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n📊 Parameter Count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  Frozen: {frozen_params:,}")
    print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")

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

    print("\n📦 Creating datasets...")
    train_dataset = SimpleDataset(train_data['texts'], train_data['labels'], tokenizer)
    val_dataset = SimpleDataset(val_data['texts'], val_data['labels'], tokenizer)
    test_dataset = SimpleDataset(test_data['texts'], test_data['labels'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ============================================================
    # 4. OPTIMIZER
    # ============================================================
    print("\n⚙️ Setting up optimizer...")

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # Higher LR for partial training
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01
    )

    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"LR: 5e-5 | Epochs: {num_epochs} | Steps/epoch: {len(train_loader)}")

    # ============================================================
    # 5. TRAINING
    # ============================================================
    print("\n🚀 Training...")
    print("This should take ~30-40 minutes")
    print("Expected final accuracy: 58-62%\n")

    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print('='*70)

        # TRAINING
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        running_loss = 0
        running_correct = 0
        running_total = 0
        print_every = 50

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            batch_correct = (predictions == labels).sum().item()
            batch_total = labels.size(0)

            epoch_correct += batch_correct
            epoch_total += batch_total
            epoch_loss += loss.item()

            running_correct += batch_correct
            running_total += batch_total
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % print_every == 0:
                running_acc = running_correct / running_total
                avg_running_loss = running_loss / print_every

                print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                      f"Loss: {avg_running_loss:.4f} | "
                      f"Acc: {running_acc:.4f} ({running_acc*100:.1f}%)")

                running_loss = 0
                running_correct = 0
                running_total = 0

        epoch_avg_loss = epoch_loss / len(train_loader)
        epoch_avg_acc = epoch_correct / epoch_total

        print(f"\n{'─'*70}")
        print(f"📊 EPOCH {epoch+1} SUMMARY:")
        print(f"   Train Loss: {epoch_avg_loss:.4f}")
        print(f"   Train Acc: {epoch_avg_acc:.4f} ({epoch_avg_acc*100:.1f}%)")
        print(f"{'─'*70}")

        # VALIDATION
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

        print(f"   Val Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"   Prediction dist: {val_dist}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   ✅ NEW BEST: {best_val_acc:.4f}")
        else:
            print(f"   📌 Best so far: {best_val_acc:.4f}")

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
    print(f"{'Language':<10} {'Accuracy':<15} {'Samples':<10}")
    print("─"*35)

    lang_accs = {}
    for lang in sorted(lang_total.keys()):
        acc = lang_correct[lang] / lang_total[lang]
        lang_accs[lang] = acc
        print(f"{lang:<10} {acc:.4f} ({acc*100:.1f}%)  {lang_total[lang]:<10}")

    avg_acc = np.mean(list(lang_accs.values()))
    print("─"*35)
    print(f"{'AVERAGE':<10} {avg_acc:.4f} ({avg_acc*100:.1f}%)")

    # ============================================================
    # 7. SAVE RESULTS
    # ============================================================
    print("\n💾 Saving results...")

    # CREATE DIRECTORY FIRST!
    import os
    os.makedirs('/content/drive/MyDrive/TADR_Results', exist_ok=True)

    results = {
        'model': 'XLM-RoBERTa Fair Baseline (Last 2 Layers)',
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'frozen_parameters': int(frozen_params),
        'overall_accuracy': float(test_acc),
        'average_accuracy': float(avg_acc),
        'language_accuracies': {k: float(v) for k, v in lang_accs.items()},
        'best_val_accuracy': float(best_val_acc),
        'prediction_distribution': {str(k): int(v) for k, v in pred_dist.items()},
        'label_distribution': {str(k): int(v) for k, v in label_dist.items()},
        'timestamp': datetime.now().isoformat()
    }

    save_path = '/content/drive/MyDrive/TADR_Results/fair_baseline_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {save_path}")

    # ============================================================
    # 8. FINAL COMPARISON
    # ============================================================
    print("\n" + "="*70)
    print("📝 THREE-WAY COMPARISON")
    print("="*70)

    print(f"\n{'Method':<25} {'Params':<15} {'Test Acc':<10}")
    print("─"*50)
    print(f"{'XLM-R Full FT':<25} {'125M':<15} {'66.22%':<10}")
    print(f"{'XLM-R Fair (Last 2)':<25} {f'{trainable_params/1e6:.1f}M':<15} {f'{test_acc*100:.2f}%':<10}")
    print(f"{'TADR (yours)':<25} {'3.8M':<15} {'66.47%':<10}")
    print("─"*50)

    tadr_acc = 0.6647
    if test_acc < tadr_acc:
        improvement = tadr_acc - test_acc
        print(f"\n✅ TADR WINS by {improvement:.4f} ({improvement*100:.1f}%)")
        print(f"   Fair baseline: {test_acc*100:.1f}%")
        print(f"   TADR:          {tadr_acc*100:.1f}%")
    else:
        print(f"\n⚠️ Fair baseline performed better than expected")
        print(f"   This is unusual but shows XLM-R is very strong")

    print("\n" + "="*70)
    print("✅ FAIR BASELINE COMPLETE!")
    print("="*70)

    return results

# ============================================================
# RUN IT NOW!
# ============================================================
print("\n🎬 Starting FAIR BASELINE")
print("Expected time: 30-40 minutes")
print("Expected result: 58-62% accuracy\n")

fair_baseline_results = train_fair_baseline()