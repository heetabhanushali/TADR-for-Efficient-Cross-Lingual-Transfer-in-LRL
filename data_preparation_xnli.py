"""
TADR Framework - XNLI Data Preparation (Zero-Shot Optimized)
Prepares data specifically for zero-shot cross-lingual transfer experiments
"""

import pandas as pd
import json
import os
from datasets import load_dataset
from typing import Dict, List, Tuple
from collections import Counter
import pickle

# ============================================================================
# LANGUAGE MAPPING
# ============================================================================

XNLI_LANG_MAPPING = {
    'ar': 'arz',  # Arabic
    'bg': 'bul',  # Bulgarian
    'de': 'deu',  # German
    'el': 'ell',  # Greek
    'en': 'eng',  # English
    'es': 'spa',  # Spanish
    'fr': 'fra',  # French
    'hi': 'hin',  # Hindi
    'ru': 'rus',  # Russian
    'sw': 'swh',  # Swahili
    'th': 'tha',  # Thai
    'tr': 'tur',  # Turkish
    'ur': 'urd',  # Urdu
    'vi': 'vie',  # Vietnamese
    'zh': 'cmn',  # Chinese
}

XNLI_LABELS = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}

# ============================================================================
# CORE FUNCTIONS (same as before)
# ============================================================================

def load_xnli_split(language: str, split: str = 'train', max_samples: int = None):
    """Load a single language split from XNLI."""
    print(f"  Loading {language} ({split})...", end=" ")
    
    try:
        dataset = load_dataset("xnli", language, trust_remote_code=True)
    except Exception as e:
        print(f"Error: {e}")
        return [], [], []
    
    texts = []
    labels = []
    iso_code = XNLI_LANG_MAPPING[language]
    
    split_data = dataset[split]
    
    if max_samples:
        split_data = split_data.select(range(min(max_samples, len(split_data))))
    
    for item in split_data:
        text = f"{item['premise']} [SEP] {item['hypothesis']}"
        texts.append(text)
        labels.append(item['label'])
    
    lang_ids = [iso_code] * len(texts)
    
    print(f"{len(texts)} samples")
    
    return texts, labels, lang_ids


def save_data_as_pickle(data: Dict, output_dir: str = './data'):
    """Save data as pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split in data.keys():
        output_path = os.path.join(output_dir, f'xnli_{split}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(data[split], f)
        print(f"  Saved: {output_path} ({len(data[split]['texts'])} samples)")


def load_preprocessed_pickle(data_dir: str = './data', split: str = 'train'):
    """Load preprocessed data from pickle."""
    pkl_path = os.path.join(data_dir, f'xnli_{split}.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def print_data_statistics(data: Dict):
    """Print detailed statistics."""
    print("\n" + "="*70)
    print("DATA STATISTICS")
    print("="*70)
    
    for split in data.keys():
        texts = data[split]['texts']
        labels = data[split]['labels']
        lang_ids = data[split]['lang_ids']
        
        print(f"\n{split.upper()}:")
        print(f"  Total samples: {len(texts)}")
        
        # Language distribution
        lang_counts = Counter(lang_ids)
        print(f"  Languages: {list(lang_counts.keys())}")
        for lang in sorted(lang_counts.keys()):
            count = lang_counts[lang]
            pct = count / len(lang_ids) * 100
            print(f"    {lang}: {count} ({pct:.1f}%)")
        
        # Label distribution
        label_counts = Counter(labels)
        print(f"  Label distribution:")
        for label_idx in sorted(label_counts.keys()):
            count = label_counts[label_idx]
            pct = count / len(labels) * 100
            print(f"    {XNLI_LABELS[label_idx]}: {count} ({pct:.1f}%)")


# ============================================================================
# ZERO-SHOT OPTIMIZED PREPARATION
# ============================================================================

def prepare_zeroshot_xnli(
    train_langs: List[str] = None,
    val_seen_langs: List[str] = None,
    val_unseen_langs: List[str] = None,
    test_langs: List[str] = None,
    train_samples_per_lang: int = 5000,
    val_samples_per_lang: int = 1000,
    test_samples_per_lang: int = 1000,
    output_dir: str = './data_zeroshot'
):
    """
    Prepare XNLI data optimized for zero-shot cross-lingual transfer.
    
    Args:
        train_langs: Languages for training (HF codes like 'en', 'hi', 'zh')
        val_seen_langs: Validation languages that appear in training
        val_unseen_langs: Validation languages NOT in training (zero-shot)
        test_langs: Test languages (completely unseen)
        train_samples_per_lang: Samples per language for training
        val_samples_per_lang: Samples per language for validation
        test_samples_per_lang: Samples per language for testing
        output_dir: Output directory
    
    Returns:
        Dictionary with prepared data
    """
    # Default configuration (recommended for zero-shot)
    if train_langs is None:
        train_langs = ['en', 'hi', 'ar']  # Diverse typology
    
    if val_seen_langs is None:
        val_seen_langs = ['en']  # Monitor training progress
    
    if val_unseen_langs is None:
        val_unseen_langs = ['sw']  # Monitor zero-shot during training
    
    if test_langs is None:
        test_langs = ['tr', 'th', 'de', 'fr']  # Final zero-shot evaluation
    
    print("\n" + "="*70)
    print("ZERO-SHOT XNLI DATA PREPARATION")
    print("="*70)
    print("\n📚 Configuration:")
    print(f"  Training languages: {train_langs}")
    print(f"  Validation (seen): {val_seen_langs}")
    print(f"  Validation (unseen/zero-shot): {val_unseen_langs}")
    print(f"  Test (unseen/zero-shot): {test_langs}")
    
    # Verify no overlap
    all_val = set(val_seen_langs + val_unseen_langs)
    all_test = set(test_langs)
    
    if not set(val_seen_langs).issubset(set(train_langs)):
        print("\n⚠️ Warning: Some validation 'seen' languages not in training set")
    
    if set(val_unseen_langs).intersection(set(train_langs)):
        print("\n⚠️ Warning: Some validation 'unseen' languages overlap with training")
    
    if all_test.intersection(set(train_langs)):
        print("\n⚠️ Warning: Some test languages overlap with training")
    
    data = {}
    
    # ============================================================================
    # TRAINING DATA (only from train_langs)
    # ============================================================================
    
    print("\n" + "-"*70)
    print("Loading TRAINING data:")
    print("-"*70)
    
    train_texts, train_labels, train_lang_ids = [], [], []
    
    for lang in train_langs:
        texts, labels, lang_ids = load_xnli_split(
            lang, 'train', train_samples_per_lang
        )
        train_texts.extend(texts)
        train_labels.extend(labels)
        train_lang_ids.extend(lang_ids)
    
    data['train'] = {
        'texts': train_texts,
        'labels': train_labels,
        'lang_ids': train_lang_ids
    }
    
    # ============================================================================
    # VALIDATION DATA (seen + unseen)
    # ============================================================================
    
    print("\n" + "-"*70)
    print("Loading VALIDATION data:")
    print("-"*70)
    
    val_texts, val_labels, val_lang_ids = [], [], []
    
    # Seen languages
    print("\n  Seen languages (in training):")
    for lang in val_seen_langs:
        texts, labels, lang_ids = load_xnli_split(
            lang, 'validation', val_samples_per_lang
        )
        val_texts.extend(texts)
        val_labels.extend(labels)
        val_lang_ids.extend(lang_ids)
    
    # Unseen languages (zero-shot)
    print("\n  Unseen languages (zero-shot):")
    for lang in val_unseen_langs:
        texts, labels, lang_ids = load_xnli_split(
            lang, 'validation', val_samples_per_lang
        )
        val_texts.extend(texts)
        val_labels.extend(labels)
        val_lang_ids.extend(lang_ids)
    
    data['validation'] = {
        'texts': val_texts,
        'labels': val_labels,
        'lang_ids': val_lang_ids
    }
    
    # ============================================================================
    # TEST DATA (all unseen for final zero-shot evaluation)
    # ============================================================================
    
    print("\n" + "-"*70)
    print("Loading TEST data (zero-shot):")
    print("-"*70)
    
    test_texts, test_labels, test_lang_ids = [], [], []
    
    for lang in test_langs:
        texts, labels, lang_ids = load_xnli_split(
            lang, 'test', test_samples_per_lang
        )
        test_texts.extend(texts)
        test_labels.extend(labels)
        test_lang_ids.extend(lang_ids)
    
    data['test'] = {
        'texts': test_texts,
        'labels': test_labels,
        'lang_ids': test_lang_ids
    }
    
    # ============================================================================
    # STATISTICS
    # ============================================================================
    
    print_data_statistics(data)
    
    # ============================================================================
    # SAVE
    # ============================================================================
    
    print("\n" + "="*70)
    print("Saving data...")
    print("="*70)
    
    save_data_as_pickle(data, output_dir)
    
    # Save configuration
    config = {
        'train_languages': train_langs,
        'val_seen_languages': val_seen_langs,
        'val_unseen_languages': val_unseen_langs,
        'test_languages': test_langs,
        'train_samples_per_lang': train_samples_per_lang,
        'val_samples_per_lang': val_samples_per_lang,
        'test_samples_per_lang': test_samples_per_lang
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Configuration saved: {config_path}")
    
    print("\n" + "="*70)
    print("✅ Zero-Shot Data Preparation Complete!")
    print("="*70)
    print(f"\nData saved to: {output_dir}")
    print(f"\nTraining setup:")
    print(f"  - Train on: {train_langs}")
    print(f"  - Validate on (seen): {val_seen_langs}")
    print(f"  - Validate on (zero-shot): {val_unseen_langs}")
    print(f"  - Test on (zero-shot): {test_langs}")
    print("\n" + "="*70 + "\n")
    
    return data


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("TADR Framework - Zero-Shot XNLI Data Preparation")
    print("🚀"*35)
    
    print("\nRecommended configuration for zero-shot transfer:")
    print("  Training: English, Hindi, Chinese (typologically diverse)")
    print("  Validation: English (seen) + Swahili (unseen)")
    print("  Test: Swahili, Arabic, Turkish, Thai (all unseen)")
    
    use_recommended = input("\nUse recommended configuration? (y/n) [default: y]: ").strip().lower()
    
    if use_recommended != 'n':
        # Recommended configuration
        data = prepare_zeroshot_xnli(
            train_langs=['en', 'hi', 'ar'],
            val_seen_langs=['en'],
            val_unseen_langs=['sw'],
            test_langs=['tr', 'th', 'de', 'fr'],
            train_samples_per_lang=5000,
            val_samples_per_lang=1000,
            test_samples_per_lang=1000,
            output_dir='./data_zeroshot'
        )
    else:
        # Custom configuration
        print("\nCustom configuration:")
        
        train_input = input("Training languages (e.g., en,hi,zh): ").strip()
        train_langs = [l.strip() for l in train_input.split(',')]
        