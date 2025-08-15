# 🚀 Quick Start Guide - Multi-Language RAG System

## ⚡ Get Running in 5 Minutes

### 1. **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Internet connection (for model downloads)

### 2. **One-Command Deployment**
```bash
# Clone or download the project
cd nervespark_assignment

# Run the automated deployment script
python deploy.py
```

### 3. **Start the System**
```bash
# Windows
start_windows.bat

# macOS/Linux
./start_unix.sh

# Or manually
source venv/bin/activate  # On Unix/Linux
streamlit run app.py
```

### 4. **Access the Web Interface**
- Open your browser
- Go to: `http://localhost:8501`
- Click "Initialize RAG Engine"
- Click "Add Sample Documents"
- Start asking questions!

## 🌍 Try These Sample Queries

### English
- "What are the symptoms of diabetes?"
- "How is diabetes treated?"
- "What causes high blood pressure?"

### Spanish
- "¿Cuáles son los síntomas de la diabetes?"
- "¿Cómo se trata la diabetes?"

### French
- "Quels sont les symptômes du diabète?"
- "Comment traite-t-on le diabète?"

### German
- "Was sind die Symptome von Diabetes?"
- "Wie wird Diabetes behandelt?"

## 🔧 Quick Configuration

### Environment Variables (Optional)
Create a `.env` file:
```env
GOOGLE_TRANSLATE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### Custom Settings
Edit `config_local.py`:
```python
CHUNK_SIZE = 800
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.8
```

## 📊 System Features

- **12 Supported Languages**: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Russian
- **Healthcare Domain**: Optimized for medical documents and terminology
- **Cross-Language Search**: Find information regardless of source language
- **Cultural Context**: Preserves cultural nuances in translations
- **Vector Database**: ChromaDB for efficient similarity search
- **Web Interface**: Beautiful Streamlit-based UI

## 🧪 Testing & Evaluation

### Run Demo
```bash
python demo.py
```

### Run Evaluation
```bash
python start_evaluation.py
```

### Test Individual Components
```python
from rag_engine import RAGEngine
from evaluation import RAGEvaluator

# Initialize and test
rag = RAGEngine()
evaluator = RAGEvaluator(rag)
results = evaluator.run_basic_evaluation()
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Download Issues**
   - Check internet connection
   - Clear cache: `rm -rf ~/.cache/huggingface/`

3. **Memory Issues**
   - Reduce `CHUNK_SIZE` in config
   - Use smaller embedding model

4. **Translation Errors**
   - Check Google Translate API key
   - Verify internet connection

### Performance Tips

- **Faster Processing**: Reduce `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **Better Accuracy**: Increase `SIMILARITY_THRESHOLD`
- **More Results**: Increase `TOP_K_RESULTS`

## 📁 Project Structure

```
nervespark_assignment/
├── app.py                 # Main Streamlit app
├── rag_engine.py         # Core RAG engine
├── document_processor.py # Document processing
├── embedding_engine.py   # Text embeddings
├── translation_service.py # Translation service
├── vector_db.py          # Vector database
├── evaluation.py         # System evaluation
├── config.py             # Configuration
├── utils.py              # Utilities
├── deploy.py             # Deployment script
├── demo.py               # Demo script
└── requirements.txt      # Dependencies
```

## 🌟 Advanced Usage

### Custom Document Types
Add support for new file formats in `document_processor.py`

### Custom Embedding Models
Change `EMBEDDING_MODEL_NAME` in config for different models

### Custom Languages
Add new languages to `SUPPORTED_LANGUAGES` in `config.py`

### API Integration
Use `RAGEngine` class directly in your Python code:

```python
from rag_engine import RAGEngine

rag = RAGEngine()
response = rag.query("Your question here", target_language="en")
print(response['answer'])
```

## 📞 Support

- **Documentation**: See `README.md` for full details
- **Issues**: Check error logs in console output
- **Configuration**: Review `config.py` and `.env` files

## 🎯 What You Can Do

1. **Process Documents**: Upload healthcare documents in any language
2. **Ask Questions**: Query in any supported language
3. **Get Answers**: Receive responses in your preferred language
4. **Track Sources**: See which documents contributed to answers
5. **Evaluate Performance**: Run comprehensive system tests
6. **Customize**: Modify settings for your specific needs

---

**🎉 You're all set! Start exploring the Multi-Language RAG System!**
