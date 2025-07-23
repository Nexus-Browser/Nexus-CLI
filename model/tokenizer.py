"""
Enhanced Tokenizer for Nexus CLI
Integrates modern tokenization with optimized encoding/decoding
Supports code-aware tokenization and special tokens
"""

import json
import regex as re
import torch
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class NexusTokenizer:
    """
    Enhanced tokenizer optimized for code generation and natural language
    Combines BPE tokenization with code-aware special tokens
    """
    
    # Special tokens for enhanced functionality
    SPECIAL_TOKENS = {
        '<|endoftext|>': 50256,
        '<|startofcode|>': 50257,
        '<|endofcode|>': 50258,
        '<|startofthought|>': 50259,
        '<|endofthought|>': 50260,
        '<|system|>': 50261,
        '<|user|>': 50262,
        '<|assistant|>': 50263,
        '<|function|>': 50264,
        '<|error|>': 50265,
        '<|debug|>': 50266,
        '<|web|>': 50267,
        '<|search|>': 50268,
        '<|result|>': 50269,
        '<|context|>': 50270,
    }
    
    # Reverse mapping
    SPECIAL_TOKENS_INV = {v: k for k, v in SPECIAL_TOKENS.items()}
    
    def __init__(self, vocab_file: str = None, merges_file: str = None):
        """Initialize tokenizer with vocab and merges files"""
        self.vocab_file = vocab_file or "tokenizer/vocab.json"
        self.merges_file = merges_file or "tokenizer/merges.txt"
        
        # Load vocabulary
        try:
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                self.encoder = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Vocab file {self.vocab_file} not found, creating minimal vocab")
            self.encoder = self._create_minimal_vocab()
        
        # Add special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            if token not in self.encoder:
                self.encoder[token] = idx
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.encoder)
        
        # Load BPE merges
        try:
            with open(self.merges_file, 'r', encoding='utf-8') as f:
                bpe_merges = f.read().strip().split('\n')[1:]  # Skip header
        except FileNotFoundError:
            logger.warning(f"Merges file {self.merges_file} not found, using basic tokenization")
            bpe_merges = []
        
        self.bpe_ranks = dict(zip([tuple(merge.split()) for merge in bpe_merges], range(len(bpe_merges))))
        
        # Compile regex patterns for efficient tokenization
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Code-aware patterns
        self.code_patterns = {
            'function_def': re.compile(r'\b(def|function|class|interface)\s+\w+'),
            'import_stmt': re.compile(r'\b(import|from|include|require)\s+'),
            'string_literal': re.compile(r'["\'].*?["\']'),
            'comment': re.compile(r'#.*?$|//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
        }
        
        logger.info(f"✓ NexusTokenizer initialized with {self.vocab_size:,} tokens")
    
    def _create_minimal_vocab(self) -> Dict[str, int]:
        """Create minimal vocabulary for fallback"""
        vocab = {}
        
        # Add basic ASCII characters
        for i in range(256):
            vocab[chr(i)] = i
        
        # Add common programming tokens
        common_tokens = [
            'def', 'class', 'import', 'from', 'if', 'else', 'for', 'while',
            'return', 'True', 'False', 'None', 'and', 'or', 'not', 'in',
            'function', 'var', 'let', 'const', 'async', 'await',
            '{', '}', '[', ']', '(', ')', ';', ':', ',', '.', '=', '+', '-'
        ]
        
        for token in common_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        return vocab
    
    def get_pairs(self, word: List[str]) -> set:
        """Get all possible pairs of adjacent symbols"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def bpe(self, token: str) -> List[str]:
        """Apply Byte Pair Encoding to a token"""
        if token in self.encoder:
            return [token]
        
        word = list(token)
        pairs = self.get_pairs(word)
        
        if not pairs:
            return [token]
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)
        
        return word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs with code-aware tokenization
        """
        if add_special_tokens:
            text = '<|startoftext|>' + text + '<|endoftext|>'
        
        # Handle special tokens first
        for special_token in self.SPECIAL_TOKENS:
            if special_token in text:
                text = text.replace(special_token, f' {special_token} ')
        
        # Split text into tokens using regex
        tokens = self.pat.findall(text)
        
        # Apply BPE to each token
        bpe_tokens = []
        for token in tokens:
            if token in self.SPECIAL_TOKENS:
                bpe_tokens.append(token)
            else:
                bpe_tokens.extend(self.bpe(token))
        
        # Convert to IDs
        ids = []
        for token in bpe_tokens:
            if token in self.encoder:
                ids.append(self.encoder[token])
            else:
                # Handle unknown tokens by encoding as bytes
                for byte in token.encode('utf-8'):
                    ids.append(byte)
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
            else:
                # Handle unknown token IDs
                tokens.append(f'<|unk_{token_id}|>')
        
        text = ''.join(tokens)
        
        # Clean up spacing around special tokens
        for special_token in self.SPECIAL_TOKENS:
            text = text.replace(f' {special_token} ', special_token)
        
        return text
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, 
                    padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts with padding and truncation
        """
        encoded_batch = []
        attention_masks = []
        
        for text in texts:
            encoded = self.encode(text)
            
            if truncation and max_length and len(encoded) > max_length:
                encoded = encoded[:max_length]
            
            attention_mask = [1] * len(encoded)
            
            if padding and max_length:
                pad_length = max_length - len(encoded)
                if pad_length > 0:
                    encoded.extend([0] * pad_length)  # Pad with 0
                    attention_mask.extend([0] * pad_length)
            
            encoded_batch.append(encoded)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(encoded_batch, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
    
    def tokenize_code(self, code: str, language: str = None) -> List[str]:
        """
        Enhanced tokenization for code with syntax awareness
        """
        tokens = []
        
        # Add language marker if provided
        if language:
            tokens.append(f'<|{language}|>')
        
        tokens.append('<|startofcode|>')
        
        # Basic code tokenization with pattern recognition
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect code patterns
            if self.code_patterns['function_def'].search(line):
                tokens.append('<|function|>')
            elif self.code_patterns['import_stmt'].search(line):
                tokens.append('<|import|>')
            
            # Tokenize the line
            line_tokens = self.encode(line, add_special_tokens=False)
            tokens.extend([self.decoder.get(tid, f'<|unk_{tid}|>') for tid in line_tokens])
            tokens.append('<|newline|>')
        
        tokens.append('<|endofcode|>')
        return tokens
    
    def create_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Create formatted chat template for conversation
        """
        formatted = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted.append(f'<|system|>{content}<|system|>')
            elif role == 'user':
                formatted.append(f'<|user|>{content}<|user|>')
            elif role == 'assistant':
                formatted.append(f'<|assistant|>{content}<|assistant|>')
            else:
                formatted.append(f'<|{role}|>{content}<|{role}|>')
        
        return ''.join(formatted)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer files to directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(save_directory, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        
        # Save merges
        merges_path = os.path.join(save_directory, 'merges.txt')
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write('#version: 0.2\n')
            for merge, rank in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(f'{merge[0]} {merge[1]}\n')
        
        # Save config
        config_path = os.path.join(save_directory, 'tokenizer_config.json')
        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.SPECIAL_TOKENS,
            'model_max_length': 2048,
            'tokenizer_class': 'NexusTokenizer'
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Tokenizer saved to {save_directory}")


def create_tokenizer(vocab_file: str = None, merges_file: str = None) -> NexusTokenizer:
    """Factory function to create enhanced tokenizer"""
    return NexusTokenizer(vocab_file, merges_file)


if __name__ == "__main__":
    # Demo the enhanced tokenizer
    tokenizer = create_tokenizer()
    
    # Test basic encoding/decoding
    text = "def hello_world():\n    print('Hello, Nexus!')\n    return True"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    logger.info(f"Original: {text}")
    logger.info(f"Encoded: {encoded[:10]}... ({len(encoded)} tokens)")
    logger.info(f"Decoded: {decoded}")
    
    # Test code tokenization
    code_tokens = tokenizer.tokenize_code(text, 'python')
    logger.info(f"Code tokens: {code_tokens[:10]}...")
    
    # Test chat template
    messages = [
        {"role": "system", "content": "You are Nexus, an intelligent coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate fibonacci."},
        {"role": "assistant", "content": "I'll create an efficient fibonacci function for you."}
    ]
    chat_formatted = tokenizer.create_chat_template(messages)
    logger.info(f"Chat template: {chat_formatted[:100]}...")
