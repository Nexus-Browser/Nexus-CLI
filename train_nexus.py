"""
Enhanced Training Script for Nexus LLM
Integrates best practices from nanoGPT and LLMs-from-scratch
Provides production-ready training with advanced optimizations
"""

import os
import time
import math
import pickle
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Model imports
from model.nexus_llm import NexusLLM, NexusConfig
from model.tokenizer import NexusTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    
    # I/O
    out_dir: str = 'model/checkpoints'
    eval_interval: int = 200
    log_interval: int = 10
    eval_iters: int = 100
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'  # 'scratch', 'resume', 'gpt2*'
    
    # Data
    dataset: str = 'custom'
    gradient_accumulation_steps: int = 5
    batch_size: int = 12
    block_size: int = 1024
    
    # Model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    
    # AdamW optimizer
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate decay
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    
    # System
    device: str = 'auto'
    dtype: str = 'bfloat16'
    compile: bool = True
    
    # Advanced optimizations
    flash_attention: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Auto-configure device and distributed settings"""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataLoader:
    """Optimized data loader for training"""
    
    def __init__(self, data_file: str, batch_size: int, block_size: int, device: str):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        # Load data
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        logger.info(f"Loaded {len(self.data):,} tokens from {data_file}")
        
        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        
    def get_batch(self, split: str = 'train') -> tuple:
        """Get a batch of data with random sampling"""
        # Sample random starting positions
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        
        # Create input and target sequences
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+1+self.block_size] for i in ix])
        
        # Move to device
        if self.device != 'cpu':
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        
        return x, y


class NexusTrainer:
    """
    Enhanced trainer combining best practices from both repositories
    Provides production-ready training with monitoring and optimization
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu'
        
        # Set up distributed training if available
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
        
        self.master_process = self.ddp_rank == 0
        
        # Create output directory
        os.makedirs(config.out_dir, exist_ok=True)
        
        # Initialize model
        self._init_model()
        
        # Initialize data loaders
        self._init_data()
        
        # Initialize optimizer and scaler
        self._init_optimizer()
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.start_time = time.time()
        
        # Mixed precision context
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=torch.float16 if config.dtype == 'float16' else torch.bfloat16
        )
        
        logger.info(f"✓ Trainer initialized on {self.device} (rank {self.ddp_rank}/{self.ddp_world_size})")
    
    def _init_model(self):
        """Initialize model with configuration"""
        model_args = {
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_embd': self.config.n_embd,
            'block_size': self.config.block_size,
            'bias': self.config.bias,
            'vocab_size': None,  # Will be set from data
            'dropout': self.config.dropout,
            'use_flash_attention': self.config.flash_attention,
            'gradient_checkpointing': self.config.gradient_checkpointing,
        }
        
        if self.config.init_from == 'scratch':
            # Initialize from scratch
            logger.info("Initializing model from scratch")
            model_args['vocab_size'] = 50304  # Default GPT-2 vocab size
            model_config = NexusConfig(**model_args)
            self.model = NexusLLM(model_config)
            
        elif self.config.init_from == 'resume':
            # Resume from checkpoint
            logger.info(f"Resuming from {self.config.out_dir}")
            ckpt_path = os.path.join(self.config.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # Load model config and state
            model_config = NexusConfig(**checkpoint['model_args'])
            self.model = NexusLLM(model_config)
            self.model.load_state_dict(checkpoint['model'])
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']
            
        elif self.config.init_from.startswith('gpt2'):
            # Initialize from GPT-2 checkpoint
            logger.info(f"Initializing from OpenAI GPT-2: {self.config.init_from}")
            override_args = {'dropout': self.config.dropout}
            self.model = NexusLLM.from_pretrained(self.config.init_from, override_args)
            
            # Crop model if needed
            if self.config.block_size < self.model.config.block_size:
                self.model.crop_block_size(self.config.block_size)
                model_args['block_size'] = self.config.block_size
        
        else:
            raise ValueError(f"Unknown init_from: {self.config.init_from}")
        
        # Move to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Compile model for 2x speedup
        if self.config.compile:
            logger.info("Compiling model with torch.compile...")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)
        
        # Wrap in DDP if distributed
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        
        # Get raw model for saving
        self.raw_model = self.model.module if self.ddp else self.model
    
    def _init_data(self):
        """Initialize data loaders"""
        data_dir = 'data'
        
        # Load training data
        train_data_path = os.path.join(data_dir, 'train.bin')
        if os.path.exists(train_data_path):
            self.train_loader = DataLoader(
                train_data_path, 
                self.config.batch_size, 
                self.config.block_size, 
                self.device
            )
        else:
            logger.warning(f"Training data not found at {train_data_path}")
            self.train_loader = None
        
        # Load validation data
        val_data_path = os.path.join(data_dir, 'val.bin')
        if os.path.exists(val_data_path):
            self.val_loader = DataLoader(
                val_data_path, 
                self.config.batch_size, 
                self.config.block_size, 
                self.device
            )
        else:
            logger.warning(f"Validation data not found at {val_data_path}")
            self.val_loader = None
    
    def _init_optimizer(self):
        """Initialize optimizer with parameter groups"""
        # Configure optimizer
        self.optimizer = self.raw_model.configure_optimizers(
            self.config.weight_decay, 
            self.config.learning_rate, 
            (self.config.beta1, self.config.beta2), 
            self.device_type
        )
        
        # Load optimizer state if resuming
        if self.config.init_from == 'resume':
            ckpt_path = os.path.join(self.config.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))
    
    def get_lr(self, it: int) -> float:
        """Learning rate schedule with warmup and cosine decay"""
        if not self.config.decay_lr:
            return self.config.learning_rate
        
        # Linear warmup
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        
        # Cosine decay
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets"""
        out = {}
        self.model.eval()
        
        for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            if loader is None:
                continue
            
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = loader.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        
        self.model.train()
        return out
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        if not self.master_process:
            return
        
        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.raw_model.config.__dict__,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }
        
        ckpt_path = os.path.join(self.config.out_dir, 'ckpt.pt')
        logger.info(f"Saving checkpoint to {ckpt_path}")
        torch.save(checkpoint, ckpt_path)
    
    def train_step(self) -> Dict[str, float]:
        """Execute one training step with gradient accumulation"""
        # Determine learning rate
        lr = self.get_lr(self.iter_num)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate and log
        if self.iter_num % self.config.eval_interval == 0 and self.master_process:
            losses = self.estimate_loss()
            logger.info(f"Step {self.iter_num}: train loss {losses.get('train', 0):.4f}, "
                       f"val loss {losses.get('val', 0):.4f}")
            
            # Save checkpoint if validation loss improved
            if losses.get('val', float('inf')) < self.best_val_loss or self.config.always_save_checkpoint:
                self.best_val_loss = losses.get('val', self.best_val_loss)
                if self.iter_num > 0:
                    self.save_checkpoint()
        
        # Training step with gradient accumulation
        self.model.train()
        total_loss = 0.0
        
        for micro_step in range(self.config.gradient_accumulation_steps):
            if self.ddp:
                # Only sync gradients on the last micro step
                self.model.require_backward_grad_sync = (micro_step == self.config.gradient_accumulation_steps - 1)
            
            with self.ctx:
                X, Y = self.train_loader.get_batch('train')
                logits, loss = self.model(X, Y)
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            total_loss += loss.item()
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        if self.iter_num % self.config.log_interval == 0 and self.master_process:
            lossf = total_loss * self.config.gradient_accumulation_steps
            dt = time.time() - self.start_time
            logger.info(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt:.2f}s, lr {lr:.2e}")
            self.start_time = time.time()
        
        self.iter_num += 1
        
        return {'loss': total_loss, 'lr': lr}
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        if self.config.eval_only:
            losses = self.estimate_loss()
            logger.info(f"Evaluation only - train: {losses.get('train', 0):.4f}, val: {losses.get('val', 0):.4f}")
            return
        
        self.start_time = time.time()
        local_iter_num = 0
        
        while True:
            # Training step
            metrics = self.train_step()
            
            # Termination conditions
            if self.iter_num > self.config.max_iters:
                break
            
            local_iter_num += 1
        
        if self.ddp:
            destroy_process_group()
        
        logger.info("Training completed!")


def prepare_data(input_file: str, output_dir: str = 'data'):
    """Prepare training data from text file"""
    logger.info(f"Preparing data from {input_file}")
    
    # Create tokenizer
    tokenizer = NexusTokenizer()
    
    # Read and tokenize data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    logger.info(f"Tokenized {len(text):,} characters into {len(tokens):,} tokens")
    
    # Split into train/validation
    n = len(tokens)
    train_data = tokens[:int(n*0.9)]
    val_data = tokens[int(n*0.9):]
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.bin'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val.bin'), 'wb') as f:
        pickle.dump(val_data, f)
    
    logger.info(f"Saved {len(train_data):,} training tokens and {len(val_data):,} validation tokens")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Nexus LLM')
    parser.add_argument('--config', type=str, help='Path to training config file')
    parser.add_argument('--data', type=str, help='Path to input text file for data preparation')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--compile', action='store_true', default=True, help='Use torch.compile')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Prepare data if specified
    if args.data:
        prepare_data(args.data)
        return
    
    # Load configuration
    config = TrainingConfig()
    
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Override with command line arguments
    if args.eval_only:
        config.eval_only = True
    if args.resume:
        config.init_from = 'resume'
    config.compile = args.compile
    
    # Create trainer and train
    trainer = NexusTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
