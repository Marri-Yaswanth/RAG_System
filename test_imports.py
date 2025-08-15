
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
