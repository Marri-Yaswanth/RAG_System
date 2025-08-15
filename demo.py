#!/usr/bin/env python3
"""
Demo Script for Multi-Language RAG System
Showcases the system's capabilities with sample queries
"""
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from rag_engine import RAGEngine
import config

def print_banner():
    """Print the demo banner."""
    print("=" * 80)
    print("🌍 MULTI-LANGUAGE RAG SYSTEM DEMO")
    print("=" * 80)
    print("Healthcare Document Intelligence Across Languages")
    print("=" * 80)
    print()

def print_supported_languages():
    """Print supported languages."""
    print("📚 SUPPORTED LANGUAGES:")
    print("-" * 40)
    
    languages = list(config.SUPPORTED_LANGUAGES.items())
    for i in range(0, len(languages), 3):
        row = languages[i:i+3]
        row_text = []
        for code, name in row:
            row_text.append(f"{name} ({code})")
        print(f"   {' | '.join(row_text)}")
    print()

def demo_healthcare_queries():
    """Demonstrate healthcare-related queries in multiple languages."""
    print("🏥 HEALTHCARE QUERY DEMONSTRATION")
    print("-" * 50)
    
    # Sample queries in different languages
    demo_queries = [
        {
            "query": "What are the symptoms of diabetes?",
            "language": "en",
            "description": "English query about diabetes symptoms"
        },
        {
            "query": "¿Cuáles son los síntomas de la diabetes?",
            "language": "es",
            "description": "Spanish query about diabetes symptoms"
        },
        {
            "query": "Quels sont les symptômes du diabète?",
            "language": "fr",
            "description": "French query about diabetes symptoms"
        },
        {
            "query": "Was sind die Symptome von Diabetes?",
            "language": "de",
            "description": "German query about diabetes symptoms"
        },
        {
            "query": "糖尿病の症状は何ですか？",
            "language": "ja",
            "description": "Japanese query about diabetes symptoms"
        }
    ]
    
    return demo_queries

def run_demo():
    """Run the main demo."""
    print_banner()
    print_supported_languages()
    
    print("🚀 Initializing RAG Engine...")
    try:
        rag_engine = RAGEngine()
        print("✅ RAG Engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG Engine: {e}")
        return False
    
    print("\n📚 Adding sample healthcare documents...")
    try:
        result = rag_engine.add_sample_documents()
        if result['status'] == 'success':
            print(f"✅ Added {result['sample_documents_added']} sample documents")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Languages: {', '.join(result['languages'])}")
        else:
            print(f"⚠️ Warning: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️ Warning: Could not add sample documents: {e}")
    
    print("\n🔍 Running demo queries...")
    demo_queries = demo_healthcare_queries()
    
    for i, query_info in enumerate(demo_queries, 1):
        print(f"\n--- Query {i}: {query_info['description']} ---")
        print(f"Query: {query_info['query']}")
        print(f"Language: {query_info['language']}")
        
        try:
            start_time = time.time()
            
            # Process query
            response = rag_engine.query(
                query_info['query'],
                target_language=query_info['language']
            )
            
            response_time = time.time() - start_time
            
            # Display results
            if response['status'] == 'success':
                print(f"✅ Status: Success")
                print(f"⏱️ Response Time: {response_time:.2f}s")
                print(f"🌐 Query Language: {response.get('query_language', 'Unknown')}")
                print(f"🌐 Response Language: {response.get('response_language', 'Unknown')}")
                print(f"📝 Answer: {response['answer'][:200]}...")
                
                if response.get('sources'):
                    print(f"📚 Sources: {len(response['sources'])} found")
                    for j, source in enumerate(response['sources'][:2]):  # Show first 2 sources
                        print(f"   Source {j+1}: Score {source['similarity_score']:.3f}")
                        print(f"      Text: {source['text'][:100]}...")
            else:
                print(f"❌ Status: {response['status']}")
                if response.get('message'):
                    print(f"   Message: {response['message']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("-" * 60)
    
    return True

def run_system_evaluation():
    """Run a quick system evaluation."""
    print("\n🧪 RUNNING SYSTEM EVALUATION")
    print("-" * 40)
    
    try:
        from evaluation import RAGEvaluator
        
        rag_engine = RAGEngine()
        evaluator = RAGEvaluator(rag_engine)
        
        print("Running basic evaluation...")
        results = evaluator.run_basic_evaluation()
        
        print(f"✅ Evaluation completed!")
        print(f"   Total Tests: {results.get('total_tests', 0)}")
        print(f"   Success Rate: {results.get('success_rate', 0):.2%}")
        print(f"   Average Response Time: {results.get('average_response_time', 0):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def main():
    """Main demo function."""
    try:
        # Run main demo
        if not run_demo():
            print("❌ Demo failed to complete")
            return
        
        # Run evaluation if requested
        print("\n" + "="*80)
        print("🎯 DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📋 WHAT WAS DEMONSTRATED:")
        print("✅ Multi-language document processing")
        print("✅ Cross-language information retrieval")
        print("✅ Healthcare domain specialization")
        print("✅ Intelligent answer generation")
        print("✅ Source tracking and relevance scoring")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Run the full web application: streamlit run app.py")
        print("2. Upload your own documents")
        print("3. Ask questions in any supported language")
        print("4. Explore the system's capabilities")
        
        print("\n🔧 TECHNICAL FEATURES:")
        print("- Sentence Transformers for multi-language embeddings")
        print("- ChromaDB for vector storage and retrieval")
        print("- Google Translate API for language translation")
        print("- Healthcare-specific text processing")
        print("- Cultural context preservation")
        
        print("\n" + "="*80)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
