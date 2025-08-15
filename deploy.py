#!/usr/bin/env python3
"""
Deployment Script for Multi-Language RAG System
Automates the setup and deployment process
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def check_dependencies():
    """Check if required system dependencies are available."""
    required_commands = ['pip3', 'git']
    
    for cmd in required_commands:
        if shutil.which(cmd) is None:
            print(f"❌ Required command '{cmd}' not found")
            return False
    
    print("✅ All required system dependencies are available")
    return True

def create_virtual_environment():
    """Create a Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔄 Creating virtual environment...")
    if run_command("python3 -m venv venv", "Creating virtual environment"):
        print("✅ Virtual environment created successfully")
        return True
    return False

def activate_virtual_environment():
    """Activate the virtual environment."""
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        activate_script = "venv/bin/activate"
    
    if os.path.exists(activate_script):
        print("✅ Virtual environment is ready")
        return True
    else:
        print("❌ Virtual environment activation script not found")
        return False

def install_dependencies():
    """Install Python dependencies."""
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip3"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "chroma_db",
        "logs",
        "data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_config_files():
    """Create configuration files if they don't exist."""
    config_files = {
        ".env": """# Environment Configuration for Multi-Language RAG System

# API Keys (Optional - system will work without these)
GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Model Configuration
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Translation Settings
DEFAULT_SOURCE_LANGUAGE=en
DEFAULT_TARGET_LANGUAGE=en

# RAG Configuration
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
""",
        "config_local.py": """# Local Configuration Overrides
# This file can be used to override default configuration values

# Example overrides:
# CHUNK_SIZE = 800
# TOP_K_RESULTS = 3
# SIMILARITY_THRESHOLD = 0.8
"""
    }
    
    for filename, content in config_files.items():
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created configuration file: {filename}")
        else:
            print(f"ℹ️ Configuration file already exists: {filename}")
    
    return True

def run_tests():
    """Run basic system tests."""
    print("🧪 Running system tests...")
    
    try:
        # Test Python imports
        test_script = """
import sys
sys.path.append('.')

try:
    import config
    import utils
    import document_processor
    import embedding_engine
    import translation_service
    import vector_db
    import rag_engine
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
        
        with open("test_imports.py", "w") as f:
            f.write(test_script)
        
        if run_command("python test_imports.py", "Testing module imports"):
            os.remove("test_imports.py")
            print("✅ Basic system tests passed")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def create_startup_scripts():
    """Create startup scripts for different platforms."""
    scripts = {
        "start_windows.bat": """@echo off
echo Starting Multi-Language RAG System...
call venv\\Scripts\\activate
streamlit run app.py
pause
""",
        "start_unix.sh": """#!/bin/bash
echo "Starting Multi-Language RAG System..."
source venv/bin/activate
streamlit run app.py
""",
        "start_evaluation.py": """#!/usr/bin/env python3
# Evaluation script
import sys
sys.path.append('.')

from evaluation import main

if __name__ == "__main__":
    main()
"""
    }
    
    for filename, content in scripts.items():
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Make Unix scripts executable
            if filename.endswith('.sh'):
                os.chmod(filename, 0o755)
            
            print(f"✅ Created startup script: {filename}")
        else:
            print(f"ℹ️ Startup script already exists: {filename}")
    
    return True

def print_deployment_info():
    """Print deployment information and next steps."""
    print("\n" + "="*60)
    print("🚀 MULTI-LANGUAGE RAG SYSTEM DEPLOYMENT COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Start the system:")
    if os.name == 'nt':  # Windows
        print("   - Run: start_windows.bat")
    else:  # Unix/Linux/macOS
        print("   - Run: ./start_unix.sh")
    
    print("\n2. Access the web interface:")
    print("   - Open your browser and go to: http://localhost:8501")
    
    print("\n3. Initialize the system:")
    print("   - Click 'Initialize RAG Engine' in the sidebar")
    print("   - Add sample documents or upload your own")
    print("   - Start asking questions in any supported language!")
    
    print("\n4. Run evaluation (optional):")
    print("   - python start_evaluation.py")
    
    print("\n🔧 CONFIGURATION:")
    print("- Edit .env file to customize settings")
    print("- Modify config_local.py for local overrides")
    
    print("\n📚 SUPPORTED LANGUAGES:")
    languages = [
        "English (en)", "Spanish (es)", "French (fr)", "German (de)",
        "Italian (it)", "Portuguese (pt)", "Chinese (zh)", "Japanese (ja)",
        "Korean (ko)", "Arabic (ar)", "Hindi (hi)", "Russian (ru)"
    ]
    
    for i in range(0, len(languages), 3):
        row = languages[i:i+3]
        print(f"   {' | '.join(row)}")
    
    print("\n🌐 FEATURES:")
    print("✅ Multi-language document processing")
    print("✅ Cross-language information retrieval")
    print("✅ Intelligent translation with cultural context")
    print("✅ Healthcare domain specialization")
    print("✅ Vector-based similarity search")
    print("✅ Streamlit web interface")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("├── app.py              # Main Streamlit application")
    print("├── rag_engine.py       # Core RAG engine")
    print("├── document_processor.py # Document processing")
    print("├── embedding_engine.py # Text embeddings")
    print("├── translation_service.py # Translation service")
    print("├── vector_db.py        # Vector database interface")
    print("├── evaluation.py       # System evaluation")
    print("├── config.py           # Configuration")
    print("├── utils.py            # Utility functions")
    print("└── requirements.txt    # Dependencies")
    
    print("\n🎯 DOMAIN FOCUS: Healthcare")
    print("The system is optimized for medical and healthcare documents")
    print("with specialized terminology handling and cultural sensitivity.")
    
    print("\n" + "="*60)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Multi-Language RAG System")
    parser.add_argument("--skip-tests", action="store_true", help="Skip system tests")
    parser.add_argument("--force", action="store_true", help="Force reinstallation")
    
    args = parser.parse_args()
    
    print("🚀 Multi-Language RAG System Deployment")
    print("=" * 50)
    
    # Pre-deployment checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Deployment steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Activating virtual environment", activate_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Creating configuration files", create_config_files),
        ("Creating startup scripts", create_startup_scripts)
    ]
    
    # Add tests if not skipped
    if not args.skip_tests:
        steps.append(("Running system tests", run_tests))
    
    # Execute deployment steps
    for step_name, step_func in steps:
        print(f"\n📋 {step_name.upper()}")
        if not step_func():
            print(f"❌ Deployment failed at: {step_name}")
            sys.exit(1)
    
    # Print deployment information
    print_deployment_info()
    
    print("\n🎉 Deployment completed successfully!")
    print("Your Multi-Language RAG System is ready to use!")

if __name__ == "__main__":
    main()
