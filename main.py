from bpe import byte_pair_encoding, encode_symbol, super_byte_pair_encoding
import requests
import time
import random
import matplotlib.pyplot as plt
from collections import Counter
import multiprocessing
import os
import pickle
import os.path

def download_text_dataset(url, max_length=None, cache_file="./data/dataset_cache.txt"):
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
    
    # compression ratio (inversion of bytes per token)
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

def visualize_tokenization(test_sentence, merges, vocab, super_merges, super_vocab):
    print("\n=== Visualizing Tokenization ===")
    encoded_bpe = encode_symbol(test_sentence, merges)
    encoded_super = encode_symbol(test_sentence, super_merges)
    
    bpe_tokens = []
    for token_id in encoded_bpe:
        if token_id in vocab:
            bpe_tokens.append(vocab[token_id])
    
    super_tokens = []
    for token_id in encoded_super:
        if token_id in super_vocab:
            super_tokens.append(super_vocab[token_id])
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 3))
    
    def draw_token_boxes(tokens, axis, color, edge_color, title):
        axis.clear()
        x_pos = 0
        for token in tokens:
            width = len(token)
            
            padding = 0.5
            total_width = width + padding
            
            axis.add_patch(plt.Rectangle((x_pos, 0), total_width, 1, 
                                        fill=True, color=color, 
                                        alpha=0.5, edgecolor=edge_color))
            
            axis.text(x_pos + total_width/2, 0.5, token, 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=10)
            
            x_pos += total_width
        
        axis.set_xlim(0, x_pos + 1)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.axis('off')
    
    draw_token_boxes(bpe_tokens, ax[0], 'lightblue', 'blue', 'BPE')
    
    draw_token_boxes(super_tokens, ax[1], 'lightpink', 'red', 'SuperBPE')
    
    plt.tight_layout()
    plt.savefig('./pngs/tokenization_visualization.png', dpi=150)
    print("Tokenization visualization saved to 'tokenization_visualization.png'")

    
def visualize_bpe_comparison(bpe_metrics, super_metrics, encoded_bpe, encoded_super, vocab, super_vocab):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.bar(['Standard BPE', 'SuperBPE'], 
            [bpe_metrics['compression_ratio'], super_metrics['compression_ratio']])
    plt.title('Compression Ratio\n(lower is better)')
    plt.ylabel('Ratio')
    
    plt.subplot(1, 3, 2)
    bpe_token_freq = sorted(Counter(encoded_bpe).values(), reverse=True)[:100]
    super_token_freq = sorted(Counter(encoded_super).values(), reverse=True)[:100]
    plt.plot(range(len(bpe_token_freq)), bpe_token_freq, label='Standard BPE')
    plt.plot(range(len(super_token_freq)), super_token_freq, label='SuperBPE')
    plt.title('Top 100 Token Frequency')
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 3, 3)
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

if __name__ == "__main__":
    dataset_url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    max_dataset_size = 500000  # characters
    
    bpe_model_path = "./models/bpe_model.pkl"
    super_bpe_model_path = "./models/super_bpe_model.pkl"
    
    text = download_text_dataset(dataset_url, max_dataset_size)
    print(f"Dataset sample: {text[:200]}...")
    print(f"Dataset size: {len(text)} characters, {len(text.encode('utf-8'))} bytes")
    
    split_point = int(len(text) * 0.8)
    train_text = text[:split_point]
    test_text = text[split_point:]
    print(f"Training set: {len(train_text)} chars, Test set: {len(test_text)} chars")
    
    os.makedirs("./models", exist_ok=True)
    
    if not (os.path.exists(bpe_model_path) and os.path.exists(super_bpe_model_path)):
        print("\n=== Training Models ===")
        num_merges = 5000
        num_threads = multiprocessing.cpu_count()
        transition_point = 256 + (num_merges // 2)
        
        print("\n=== Training Standard BPE ===")
        start_time = time.time()
        tokenized, vocab, merges = byte_pair_encoding(train_text, num_merges, num_threads)
        bpe_training_time = time.time() - start_time
        print(f"BPE Training time: {bpe_training_time:.2f} seconds")
        
        with open(bpe_model_path, 'wb') as f:
            pickle.dump((vocab, merges), f)
        
        print("\n=== Training SuperBPE ===")
        start_time = time.time()
        super_tokenized, super_vocab, super_merges = super_byte_pair_encoding(
            train_text, num_merges, transition_point, num_threads
        )
        super_bpe_training_time = time.time() - start_time
        print(f"SuperBPE Training time: {super_bpe_training_time:.2f} seconds")
        
        with open(super_bpe_model_path, 'wb') as f:
            pickle.dump((super_vocab, super_merges), f)
            
        print("\n=== Models saved to disk ===")
    else:
        print("\n=== Loading existing models ===")
        with open(bpe_model_path, 'rb') as f:
            vocab, merges = pickle.load(f)
        with open(super_bpe_model_path, 'rb') as f:
            super_vocab, super_merges = pickle.load(f)
    
    print("\n=== Encoding Test Data ===")
    encoded_bpe = encode_symbol(test_text, merges)
    encoded_super = encode_symbol(test_text, super_merges)
    
    print("\n=== Evaluating Models ===")
    metrics_to_evaluate = ["compression_ratio", "unique_tokens", "avg_token_length", "decoding_accuracy"]
    
    bpe_metrics = evaluate_encoding(test_text, encoded_bpe, vocab)
    super_metrics = evaluate_encoding(test_text, encoded_super, super_vocab)
    
    if not (os.path.exists(bpe_model_path) and os.path.exists(super_bpe_model_path)):
        print("\n=== Comparative Analysis ===")
        print(f"{'Metric':<25} {'Standard BPE':<15} {'SuperBPE':<15} {'Difference (%)':<15}")
        print("-" * 70)
        
        for metric in metrics_to_evaluate:
            bpe_val = bpe_metrics[metric]
            super_val = super_metrics[metric]
            diff_pct = ((super_val - bpe_val) / bpe_val) * 100 if bpe_val else 0
            print(f"{metric:<25} {bpe_val:<15.4f} {super_val:<15.4f} {diff_pct:<15.2f}")
        
        print(f"{'Training time':<25} {bpe_training_time:<15.2f} {super_bpe_training_time:<15.2f} "
              f"{((super_bpe_training_time - bpe_training_time) / bpe_training_time) * 100:<15.2f}")
        
        superword_tokens = sum(1 for token_id, token_str in super_vocab.items() 
                             if token_id >= 256 and ' ' in token_str)
        print(f"\nNumber of superword tokens in SuperBPE vocabulary: {superword_tokens}")
        print(f"Percentage of superword tokens: {(superword_tokens / len(super_vocab)) * 100:.2f}%")
    
    print("\n=== Generating Visualizations ===")
    os.makedirs("./pngs", exist_ok=True)

    test_sentence = "This is a test sentence to compare both BPE and SuperBPE."
    visualize_tokenization(test_sentence, merges, vocab, super_merges, super_vocab)

    visualize_bpe_comparison(bpe_metrics, super_metrics, encoded_bpe, encoded_super, vocab, super_vocab)