#!/usr/bin/env python3
"""
Enhanced Nexus CLI Setup Script
Automated installation and configuration with dependency checking
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NexusSetup:
    """Automated setup for enhanced Nexus CLI"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.system = platform.system()
        self.architecture = platform.machine()
        self.project_root = Path(__file__).parent
        
    def check_python_version(self):
        """Ensure Python 3.8+ is being used"""
        logger.info(f"Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        if self.python_version < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        if self.python_version >= (3, 12):
            logger.warning("Python 3.12+ detected - some packages may have compatibility issues")
        
        return True
    
    def check_system_requirements(self):
        """Check system-specific requirements"""
        logger.info(f"System: {self.system} {self.architecture}")
        
        # Check for CUDA if available
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ NVIDIA GPU detected")
                return True
        except FileNotFoundError:
            pass
        
        # Check for Apple Silicon optimizations
        if self.system == "Darwin" and "arm" in self.architecture.lower():
            logger.info("✓ Apple Silicon detected - will use optimized PyTorch")
        
        logger.info("No GPU detected - will use CPU mode")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'data',
            'model/checkpoints',
            'memory',
            'logs',
            'tokenizer',
            'cache'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")
    
    def install_dependencies(self):
        """Install Python dependencies with optimizations"""
        logger.info("Installing dependencies...")
        
        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        
        # Install core dependencies
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            logger.info("Installing from requirements.txt...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '-r', str(requirements_file)
            ], check=True)
        else:
            logger.warning("requirements.txt not found, installing minimal dependencies")
            minimal_deps = [
                'torch>=2.1.0',
                'transformers>=4.36.0', 
                'numpy>=1.24.0',
                'requests>=2.31.0',
                'rich>=13.6.0'
            ]
            subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + minimal_deps, check=True)
        
        # Install system-specific optimizations
        self._install_system_optimizations()
        
        logger.info("✓ Dependencies installed successfully")
    
    def _install_system_optimizations(self):
        """Install system-specific optimizations"""
        
        # Try to install FlashAttention if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Installing FlashAttention for CUDA acceleration...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 
                    'flash-attn', '--no-build-isolation'
                ], check=False)  # Don't fail if this doesn't work
        except ImportError:
            pass
        
        # Install Triton if on Linux/CUDA
        if self.system == "Linux":
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'triton'
                ], check=False)
            except:
                pass
    
    def setup_configuration(self):
        """Create default configuration files"""
        
        # Model configuration
        model_config = {
            "block_size": 2048,
            "vocab_size": 50304,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "dropout": 0.0,
            "bias": False,
            "use_flash_attention": True,
            "use_kv_cache": True,
            "temperature": 0.8,
            "top_k": 200,
            "max_new_tokens": 500
        }
        
        config_path = self.project_root / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info("✓ Created model configuration")
        
        # Training configuration
        training_config = {
            "batch_size": 8,
            "learning_rate": 6e-4,
            "max_iters": 10000,
            "eval_interval": 200,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.1,
            "compile": True,
            "flash_attention": True
        }
        
        train_config_path = self.project_root / 'training_config.json'
        with open(train_config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        logger.info("✓ Created training configuration")
        
        # Environment configuration
        env_example = """# Nexus CLI Environment Configuration
# Copy this file to .env and customize as needed

# Model settings
NEXUS_MODEL_PATH=model/nexus_model
NEXUS_CONFIG_PATH=model_config.json
NEXUS_DEVICE=auto

# API Keys (optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here

# Web Intelligence
WEB_SEARCH_ENABLED=true
WEB_CACHE_DURATION=3600

# Performance
TORCH_COMPILE=true
FLASH_ATTENTION=true
KV_CACHE=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/nexus.log
"""
        
        env_path = self.project_root / '.env.example'
        with open(env_path, 'w') as f:
            f.write(env_example)
        logger.info("✓ Created environment example")
    
    def create_startup_scripts(self):
        """Create convenient startup scripts"""
        
        # Unix/Mac startup script
        if self.system in ["Linux", "Darwin"]:
            startup_script = f"""#!/bin/bash
# Nexus CLI Startup Script

cd "{self.project_root}"

echo "🚀 Starting Nexus CLI..."
echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model is available
if [ ! -f "model/nexus_model/model.pt" ]; then
    echo "⚠️  No trained model found. Run 'python train_nexus.py --data your_data.txt' first."
fi

# Start the CLI
python nexus.py --interactive

"""
            script_path = self.project_root / 'start_nexus.sh'
            with open(script_path, 'w') as f:
                f.write(startup_script)
            script_path.chmod(0o755)
            logger.info("✓ Created Unix startup script")
        
        # Windows batch script
        if self.system == "Windows":
            batch_script = f"""@echo off
REM Nexus CLI Startup Script

cd /d "{self.project_root}"

echo 🚀 Starting Nexus CLI...
echo Python version:
python --version
echo Working directory: %cd%
echo.

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Check if model is available
if not exist "model\\nexus_model\\model.pt" (
    echo ⚠️  No trained model found. Run 'python train_nexus.py --data your_data.txt' first.
)

REM Start the CLI
python nexus.py --interactive

pause
"""
            script_path = self.project_root / 'start_nexus.bat'
            with open(script_path, 'w') as f:
                f.write(batch_script)
            logger.info("✓ Created Windows startup script")
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        logger.info("Running installation tests...")
        
        try:
            # Test core imports
            import torch
            import transformers
            import numpy as np
            logger.info("✓ Core dependencies imported successfully")
            
            # Test PyTorch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            test_tensor = torch.randn(2, 3).to(device)
            logger.info(f"✓ PyTorch working on {device}")
            
            # Test model imports
            sys.path.append(str(self.project_root / 'model'))
            from nexus_llm import NexusConfig, NexusLLM
            from tokenizer import NexusTokenizer
            logger.info("✓ Nexus model components imported successfully")
            
            # Test CLI import
            sys.path.append(str(self.project_root))
            # Don't actually import nexus.py as it may start the CLI
            logger.info("✓ All components verified")
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
    
    def print_success_message(self):
        """Print success message with usage instructions"""
        print("\n" + "="*60)
        print("🎉 NEXUS CLI SETUP COMPLETE!")
        print("="*60)
        print()
        print("🚀 Quick Start:")
        
        if self.system in ["Linux", "Darwin"]:
            print("   ./start_nexus.sh")
        else:
            print("   start_nexus.bat")
        
        print("   OR")
        print("   python nexus.py --interactive")
        print()
        print("📚 Documentation:")
        print("   - README.md for detailed usage")
        print("   - model_config.json for model settings")
        print("   - .env.example for environment configuration")
        print()
        print("🔧 Training your own model:")
        print("   python train_nexus.py --data your_training_data.txt")
        print()
        print("📊 Performance Tips:")
        print("   - Use CUDA GPU for 10x faster inference")
        print("   - Enable torch.compile for 2x speedup")
        print("   - Use FlashAttention for memory efficiency")
        print()
        print("Need help? Check the documentation or run with --help")
        print("="*60)
    
    def run_setup(self):
        """Run the complete setup process"""
        logger.info("Starting Enhanced Nexus CLI setup...")
        
        if not self.check_python_version():
            sys.exit(1)
        
        self.check_system_requirements()
        self.create_directories()
        self.install_dependencies()
        self.setup_configuration()
        self.create_startup_scripts()
        
        if self.run_tests():
            self.print_success_message()
            return True
        else:
            logger.error("Setup completed with errors - some features may not work")
            return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Nexus CLI Setup')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip installation tests')
    parser.add_argument('--minimal', action='store_true', 
                       help='Minimal installation without optional dependencies')
    
    args = parser.parse_args()
    
    setup = NexusSetup()
    
    try:
        if not setup.check_python_version():
            sys.exit(1)
        
        setup.check_system_requirements()
        setup.create_directories()
        
        if not args.skip_deps:
            setup.install_dependencies()
        
        setup.setup_configuration()
        setup.create_startup_scripts()
        
        if not args.skip_tests:
            test_success = setup.run_tests()
            if not test_success:
                logger.warning("Some tests failed - check the installation")
        
        setup.print_success_message()
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
