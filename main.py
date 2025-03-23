from bpe import byte_pair_encoding, encode_symbol, super_byte_pair_encoding
import requests
import time
import random
import matplotlib.pyplot as plt
from collections import Counter
import multiprocessing
import os

def download_text_dataset(url, max_length=None, cache_file="dataset_cache.txt"):
    try:
        if os.path.exists(cache_file):
            print(f"Loading dataset from local cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                text = f.read()
                
            if max_length and len(text) > max_length:
                print(f"Truncating text to {max_length} characters")
                text = text[:max_length]
                
            print(f"Loaded dataset size: {len(text)} characters")
            return text
                
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        
        print(f"Saving dataset to local cache: {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        if max_length and len(text) > max_length:
            print(f"Truncating text to {max_length} characters")
            text = text[:max_length]
            
        print(f"Downloaded dataset size: {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error downloading/loading dataset: {e}")
        return "This is a test. This is another test. Testing testing testing is important for software development."

def evaluate_encoding(original_text, encoded_tokens, vocab):
    encoding_time_start = time.time()
    
    # compression ratio
    original_bytes = len(original_text.encode('utf-8'))
    compression_ratio = len(encoded_tokens) / original_bytes
    
    # token frequency distribution
    token_counts = Counter(encoded_tokens)
    unique_tokens = len(token_counts)
    
    # average token length in characters
    avg_token_length = sum(len(vocab.get(token, "")) for token in encoded_tokens) / len(encoded_tokens)
    
    # decoding accuracy on a sample
    sample_length = min(1000, len(encoded_tokens))
    sample_start = random.randint(0, len(encoded_tokens) - sample_length)
    sample_tokens = encoded_tokens[sample_start:sample_start + sample_length]
    
    decoded_text = ""
    for token_id in sample_tokens:
        if token_id in vocab:
            decoded_text += vocab[token_id]
    
    original_sample = original_text[len(''.join([vocab.get(t, "") for t in encoded_tokens[:sample_start]])):]
    original_sample = original_sample[:len(decoded_text)]
    
    accuracy = sum(1 for a, b in zip(original_sample, decoded_text) if a == b) / len(original_sample) if original_sample else 0
    
    encoding_time = time.time() - encoding_time_start
    
    return {
        "compression_ratio": compression_ratio,
        "unique_tokens": unique_tokens,
        "avg_token_length": avg_token_length,
        "decoding_accuracy": accuracy,
        "encoding_time": encoding_time
    }

if __name__ == "__main__":
    dataset_url = "https://www.gutenberg.org/files/1342/1342-0.txt"

    max_dataset_size = 500000  # characters
    
    text = download_text_dataset(dataset_url, max_dataset_size)
    print(f"Dataset sample: {text[:200]}...")
    print(f"Dataset size: {len(text)} characters, {len(text.encode('utf-8'))} bytes")
    
    num_merges = 5000
    num_threads = multiprocessing.cpu_count() # can change this to a number like 4 if needed
    transition_point = num_merges // 2  # transition point for SuperBPE
    
    split_point = int(len(text) * 0.8)
    train_text = text[:split_point]
    test_text = text[split_point:]
    print(f"Training set: {len(train_text)} chars, Test set: {len(test_text)} chars")
    
    print("\n=== Training Standard BPE ===")
    start_time = time.time()
    tokenized, vocab, merges = byte_pair_encoding(train_text, num_merges, num_threads)
    bpe_training_time = time.time() - start_time
    print(f"BPE Training time: {bpe_training_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    
    print("\n=== Training SuperBPE ===")
    start_time = time.time()
    super_tokenized, super_vocab, super_merges = super_byte_pair_encoding(
        train_text, num_merges, transition_point, num_threads
    )
    super_bpe_training_time = time.time() - start_time
    print(f"SuperBPE Training time: {super_bpe_training_time:.2f} seconds")
    print(f"Vocabulary size: {len(super_vocab)}")
    
    print("\n=== Encoding Test Data ===")
    start_time = time.time()
    encoded_bpe = encode_symbol(test_text, merges)
    bpe_encoding_time = time.time() - start_time
    
    start_time = time.time()
    encoded_super = encode_symbol(test_text, super_merges)
    super_encoding_time = time.time() - start_time
    
    print(f"BPE encoding time: {bpe_encoding_time:.2f} seconds")
    print(f"SuperBPE encoding time: {super_encoding_time:.2f} seconds")
    
    print("\n=== Evaluating Models ===")
    bpe_metrics = evaluate_encoding(test_text, encoded_bpe, vocab)
    super_metrics = evaluate_encoding(test_text, encoded_super, super_vocab)
    
    print("\n=== Comparative Analysis ===")
    print(f"{'Metric':<25} {'Standard BPE':<15} {'SuperBPE':<15} {'Difference (%)':<15}")
    print("-" * 70)
    
    for metric in ["compression_ratio", "unique_tokens", "avg_token_length", "decoding_accuracy", "encoding_time"]:
        bpe_val = bpe_metrics[metric]
        super_val = super_metrics[metric]
        diff_pct = ((super_val - bpe_val) / bpe_val) * 100 if bpe_val else 0
        
        print(f"{metric:<25} {bpe_val:<15.4f} {super_val:<15.4f} {diff_pct:<15.2f}")
    
    print(f"{'Training time':<25} {bpe_training_time:<15.2f} {super_bpe_training_time:<15.2f} "
          f"{((super_bpe_training_time - bpe_training_time) / bpe_training_time) * 100:<15.2f}")
    
    superword_tokens = 0
    for token_id, token_str in super_vocab.items():
        if token_id >= 256 and ' ' in token_str:
            superword_tokens += 1
    
    print(f"\nNumber of superword tokens in SuperBPE vocabulary: {superword_tokens}")
    print(f"Percentage of superword tokens: {(superword_tokens / len(super_vocab)) * 100:.2f}%")
    
    print("\n=== Generating Visualizations ===")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar(['Standard BPE', 'SuperBPE'], [bpe_metrics['compression_ratio'], super_metrics['compression_ratio']])
    plt.title('Compression Ratio (lower is better)')
    plt.ylabel('Ratio')
    
    plt.subplot(2, 2, 2)
    
    bpe_token_freq = sorted(Counter(encoded_bpe).values(), reverse=True)[:100]
    super_token_freq = sorted(Counter(encoded_super).values(), reverse=True)[:100]
    
    plt.plot(range(len(bpe_token_freq)), bpe_token_freq, label='Standard BPE')
    plt.plot(range(len(super_token_freq)), super_token_freq, label='SuperBPE')
    plt.title('Top 100 Token Frequency Distribution')
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.bar(['BPE Train', 'SuperBPE Train', 'BPE Encode', 'SuperBPE Encode'], 
            [bpe_training_time, super_bpe_training_time, bpe_encoding_time, super_encoding_time])
    plt.title('Processing Time (seconds)')
    plt.ylabel('Seconds')
    
    plt.subplot(2, 2, 4)
    
    bpe_lengths = [len(vocab.get(t, "")) for t in set(encoded_bpe)]
    super_lengths = [len(super_vocab.get(t, "")) for t in set(encoded_super)]
    
    plt.hist(bpe_lengths, alpha=0.5, label='Standard BPE')
    plt.hist(super_lengths, alpha=0.5, label='SuperBPE')
    plt.title('Token Length Distribution')
    plt.xlabel('Token Length (chars)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./pngs/bpe_comparison.png')
    print("Visualizations saved to 'bpe_comparison.png'")