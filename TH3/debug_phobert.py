#!/usr/bin/env python
"""
Debug script to find and fix out-of-bounds token IDs in PhoBERT training
"""

import torch
from transformers import PhobertTokenizer, AutoConfig
from datasets import load_dataset
from collections import defaultdict
import statistics

def debug_tokenization():
    """Check for invalid token IDs in the dataset"""
    
    # Load tokenizer
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model vocab size: {config.vocab_size}")
    print(f"Max position embeddings: {config.max_position_embeddings}")
    
    # Load a sample from your dataset
    raw_datasets = load_dataset(
        "json",
        data_files={"train": "C:/E old/DS310/DS310/TH3/dataset/train.json"},  # Replace with your file
        field="data"
    )
    
    # Test tokenization
    sample = raw_datasets["train"][0]
    question = sample["question"]
    context = sample["context"]
    
    # Tokenize
    encoded = tokenizer.encode_plus(
        question,
        context,
        max_length=256,  # Use safe length
        truncation=True,
        return_tensors="pt"
    )
    
    # Check for out-of-bounds IDs
    input_ids = encoded["input_ids"][0]
    max_id = input_ids.max().item()
    min_id = input_ids.min().item()
    
    print(f"\nToken ID range: {min_id} to {max_id}")
    print(f"Any out of bounds? {max_id >= tokenizer.vocab_size}")
    
    if max_id >= tokenizer.vocab_size:
        print("\n❌ FOUND OUT-OF-BOUNDS TOKEN IDs!")
        invalid_positions = (input_ids >= tokenizer.vocab_size).nonzero(as_tuple=True)[0]
        print(f"Invalid token positions: {invalid_positions.tolist()}")
        print(f"Invalid token IDs: {input_ids[invalid_positions].tolist()}")
    else:
        print("\n✓ All token IDs are valid")
    
    return tokenizer, encoded


def test_tokenization_batch(tokenizer, dataset, num_samples=None, max_length=256):
    """
    Test tokenization on multiple samples from the dataset
    
    Args:
        tokenizer: PhoBERT tokenizer
        dataset: Dataset to test
        num_samples: Number of samples to test (None = all)
        max_length: Maximum sequence length
    """
    print("\n" + "="*60)
    print("BATCH TOKENIZATION TEST")
    print("="*60)
    
    total_samples = len(dataset)
    test_samples = min(num_samples, total_samples) if num_samples else total_samples
    
    print(f"Testing {test_samples} samples from {total_samples} total samples...")
    
    stats = {
        'total_tested': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'token_lengths': [],
        'max_ids': [],
        'min_ids': [],
        'invalid_token_ids': [],
        'samples_with_errors': []
    }
    
    for i in range(test_samples):
        try:
            sample = dataset[i]
            question = sample.get("question", "")
            context = sample.get("context", "")
            
            # Tokenize
            encoded = tokenizer.encode_plus(
                question,
                context,
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"][0]
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            seq_length = len(input_ids)
            
            stats['total_tested'] += 1
            stats['token_lengths'].append(seq_length)
            stats['max_ids'].append(max_id)
            stats['min_ids'].append(min_id)
            
            # Check for out-of-bounds
            if max_id >= tokenizer.vocab_size:
                stats['invalid_samples'] += 1
                invalid_positions = (input_ids >= tokenizer.vocab_size).nonzero(as_tuple=True)[0]
                invalid_ids = input_ids[invalid_positions].tolist()
                stats['invalid_token_ids'].extend(invalid_ids)
                stats['samples_with_errors'].append({
                    'index': i,
                    'invalid_positions': invalid_positions.tolist(),
                    'invalid_ids': invalid_ids,
                    'max_id': max_id,
                    'question': question[:50] + "..." if len(question) > 50 else question
                })
            else:
                stats['valid_samples'] += 1
                
        except Exception as e:
            print(f"\n⚠️  Error processing sample {i}: {str(e)}")
            stats['samples_with_errors'].append({
                'index': i,
                'error': str(e)
            })
    
    # Print statistics
    print_statistics(stats, tokenizer)
    
    return stats


def print_statistics(stats, tokenizer):
    """Print detailed statistics from testing"""
    print("\n" + "-"*60)
    print("TEST RESULTS SUMMARY")
    print("-"*60)
    
    print(f"\n📊 Sample Statistics:")
    print(f"  Total tested: {stats['total_tested']}")
    print(f"  Valid samples: {stats['valid_samples']} ({stats['valid_samples']/stats['total_tested']*100:.2f}%)")
    print(f"  Invalid samples: {stats['invalid_samples']} ({stats['invalid_samples']/stats['total_tested']*100:.2f}%)")
    
    if stats['token_lengths']:
        print(f"\n📏 Token Length Statistics:")
        print(f"  Min length: {min(stats['token_lengths'])}")
        print(f"  Max length: {max(stats['token_lengths'])}")
        print(f"  Mean length: {statistics.mean(stats['token_lengths']):.2f}")
        print(f"  Median length: {statistics.median(stats['token_lengths'])}")
    
    if stats['max_ids']:
        print(f"\n🔢 Token ID Statistics:")
        print(f"  Min token ID: {min(stats['min_ids'])}")
        print(f"  Max token ID: {max(stats['max_ids'])}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Out of bounds: {max(stats['max_ids']) >= tokenizer.vocab_size}")
    
    if stats['invalid_token_ids']:
        print(f"\n❌ Invalid Token IDs Found:")
        invalid_counts = defaultdict(int)
        for invalid_id in stats['invalid_token_ids']:
            invalid_counts[invalid_id] += 1
        print(f"  Total invalid tokens: {len(stats['invalid_token_ids'])}")
        print(f"  Unique invalid IDs: {len(invalid_counts)}")
        print(f"  Most common invalid IDs:")
        for invalid_id, count in sorted(invalid_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    ID {invalid_id}: {count} occurrences")
    
    if stats['samples_with_errors']:
        print(f"\n⚠️  Samples with Errors ({len(stats['samples_with_errors'])}):")
        for error_info in stats['samples_with_errors'][:5]:  # Show first 5
            if 'error' in error_info:
                print(f"  Sample {error_info['index']}: {error_info['error']}")
            else:
                print(f"  Sample {error_info['index']}: {len(error_info['invalid_ids'])} invalid tokens")
                print(f"    Question: {error_info['question']}")
                print(f"    Max ID: {error_info['max_id']} (vocab size: {tokenizer.vocab_size})")
        if len(stats['samples_with_errors']) > 5:
            print(f"  ... and {len(stats['samples_with_errors']) - 5} more")
    
    print("\n" + "="*60)


def test_edge_cases(tokenizer):
    """Test edge cases for tokenization"""
    print("\n" + "="*60)
    print("EDGE CASE TESTING")
    print("="*60)
    
    edge_cases = [
        ("", ""),  # Empty strings
        ("Short", "Short context"),  # Very short
        ("A" * 1000, "B" * 1000),  # Very long
        ("Câu hỏi tiếng Việt?", "Ngữ cảnh tiếng Việt với nhiều ký tự đặc biệt: !@#$%^&*()"),  # Vietnamese with special chars
        ("Question with\nnewlines\tand\ttabs", "Context with\nmultiple\nlines"),  # Newlines and tabs
        ("Question with unicode: 🚀 📚", "Context with emoji: 😊 🎉"),  # Unicode/emoji
    ]
    
    print(f"\nTesting {len(edge_cases)} edge cases...")
    
    for i, (question, context) in enumerate(edge_cases):
        try:
            encoded = tokenizer.encode_plus(
                question,
                context,
                max_length=256,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"][0]
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            
            is_valid = max_id < tokenizer.vocab_size
            status = "✓" if is_valid else "❌"
            
            print(f"\n{status} Edge case {i+1}:")
            print(f"  Question length: {len(question)} chars")
            print(f"  Context length: {len(context)} chars")
            print(f"  Token length: {len(input_ids)}")
            print(f"  Token ID range: {min_id} to {max_id}")
            print(f"  Valid: {is_valid}")
            
            if not is_valid:
                invalid_positions = (input_ids >= tokenizer.vocab_size).nonzero(as_tuple=True)[0]
                print(f"  Invalid positions: {len(invalid_positions)}")
                
        except Exception as e:
            print(f"\n⚠️  Edge case {i+1} failed: {str(e)}")
    
    print("\n" + "="*60)


def run_all_tests(num_samples=100):
    """Run all tests"""
    print("="*60)
    print("PHOBERT TOKENIZATION TEST SUITE")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    
    print(f"✓ Vocab size: {tokenizer.vocab_size}")
    print(f"✓ Model vocab size: {config.vocab_size}")
    print(f"✓ Max position embeddings: {config.max_position_embeddings}")
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        raw_datasets = load_dataset(
            "json",
            data_files={"train": "C:/E old/DS310/DS310/TH3/dataset/train.json"},
            field="data"
        )
        print(f"✓ Dataset loaded: {len(raw_datasets['train'])} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {str(e)}")
        return
    
    # Run single sample test
    print("\n" + "="*60)
    print("SINGLE SAMPLE TEST")
    print("="*60)
    debug_tokenization()
    
    # Run batch test
    test_tokenization_batch(tokenizer, raw_datasets["train"], num_samples=num_samples)
    
    # Run edge case test
    test_edge_cases(tokenizer)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run all tests or just single debug
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_all_tests(num_samples=num_samples)
    else:
        debug_tokenization()