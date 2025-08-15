"""
Multi-Language RAG System - Streamlit Web Application
Provides a user-friendly interface for document processing and querying
"""
import streamlit as st
import os
import time
import pandas as pd
from pathlib import Path
import tempfile
import shutil

import config
from rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="Multi-Language RAG System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'sample_docs_added' not in st.session_state:
        st.session_state.sample_docs_added = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def initialize_rag_engine():
    """Initialize the RAG engine."""
    try:
        with st.spinner("Initializing RAG Engine..."):
            st.session_state.rag_engine = RAGEngine()
        st.success("RAG Engine initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {str(e)}")
        return False

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🌍 Multi-Language RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Healthcare Document Intelligence Across Languages</p>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Initialize RAG Engine button
        if st.button("🚀 Initialize RAG Engine", use_container_width=True):
            if initialize_rag_engine():
                st.session_state.documents_loaded = False
        
        # System Status
        if st.session_state.rag_engine:
            st.success("✅ RAG Engine Active")
            
            # Add sample documents
            if not st.session_state.sample_docs_added:
                if st.button("📚 Add Sample Documents", use_container_width=True):
                    with st.spinner("Adding sample healthcare documents..."):
                        result = st.session_state.rag_engine.add_sample_documents()
                        if result['status'] == 'success':
                            st.session_state.sample_docs_added = True
                            st.success("Sample documents added successfully!")
                        else:
                            st.error(f"Failed to add sample documents: {result.get('message', 'Unknown error')}")
            
            # System information
            st.subheader("📊 System Info")
            system_status = st.session_state.rag_engine.get_system_status()
            
            st.metric("Languages", system_status.get('system_config', {}).get('supported_languages', 0))
            st.metric("Chunk Size", system_status.get('system_config', {}).get('chunk_size', 0))
            st.metric("Top K Results", system_status.get('system_config', {}).get('top_k_results', 0))
            
            # Database info
            if 'vector_database' in system_status:
                db_info = system_status['vector_database']
                if 'collection_stats' in db_info:
                    stats = db_info['collection_stats']
                    st.metric("Documents", stats.get('total_documents', 0))
                    st.metric("Chunks", stats.get('total_chunks', 0))
        else:
            st.warning("⚠️ RAG Engine not initialized")
        
        # Language selection
        st.subheader("🌐 Response Language")
        response_language = st.selectbox(
            "Choose response language:",
            options=list(config.SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: config.SUPPORTED_LANGUAGES[x],
            index=0
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            top_k = st.slider("Top K Results", 1, 10, config.TOP_K_RESULTS)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, config.SIMILARITY_THRESHOLD, 0.1)
            include_sources = st.checkbox("Include Sources", value=True)
    
    # Main content area
    if not st.session_state.rag_engine:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ## Welcome to the Multi-Language RAG System!
        
        This system provides intelligent document retrieval and question answering across multiple languages,
        with a focus on healthcare applications.
        
        **To get started:**
        1. Click "Initialize RAG Engine" in the sidebar
        2. Add sample documents or upload your own
        3. Start asking questions in any supported language
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display supported languages
        st.subheader("🌍 Supported Languages")
        col1, col2, col3, col4 = st.columns(4)
        
        languages = list(config.SUPPORTED_LANGUAGES.items())
        for i, (code, name) in enumerate(languages):
            col = [col1, col2, col3, col4][i % 4]
            col.metric(name, code.upper())
        
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📁 Document Management", "📊 Analytics", "🧪 Testing"])
    
    with tab1:
        st.header("💬 Multi-Language Chat Interface")
        
        # Chat input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question in any language:",
                placeholder="e.g., What are the symptoms of diabetes? / ¿Cuáles son los síntomas de la diabetes? / Quels sont les symptômes du diabète?"
            )
        
        with col2:
            ask_button = st.button("🔍 Ask", use_container_width=True)
        
        # Process question
        if ask_button and user_question:
            if not st.session_state.documents_loaded and not st.session_state.sample_docs_added:
                st.warning("⚠️ Please add some documents first (use sample documents or upload your own)")
            else:
                with st.spinner("Processing your question..."):
                    # Get response from RAG engine
                    response = st.session_state.rag_engine.query(
                        user_question,
                        target_language=response_language,
                        top_k=top_k,
                        include_sources=include_sources
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'response': response,
                        'timestamp': time.time()
                    })
                    
                    # Display response
                    if response['status'] == 'success':
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"**Answer:** {response['answer']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        if response.get('sources') and include_sources:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(response['sources']):
                                with st.expander(f"Source {i+1} (Score: {source['similarity_score']:.3f})"):
                                    st.write(f"**Text:** {source['text']}")
                                    if source.get('metadata'):
                                        st.write(f"**Language:** {source['metadata'].get('language', 'Unknown')}")
                                        st.write(f"**File:** {source['metadata'].get('filename', 'Unknown')}")
                        
                        # Display metadata
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Query Language", response.get('query_language', 'Unknown'))
                        col2.metric("Response Language", response.get('response_language', 'Unknown'))
                        col3.metric("Processing Time", f"{response.get('processing_time', 0):.2f}s")
                        
                    elif response['status'] == 'no_results':
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.warning(response['answer'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(response.get('message', 'An error occurred'))
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat history
        if st.session_state.chat_history:
            st.subheader("💭 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.write(f"**Question:** {chat['question']}")
                    if chat['response']['status'] == 'success':
                        st.write(f"**Answer:** {chat['response']['answer']}")
                        st.write(f"**Language:** {chat['response'].get('query_language', 'Unknown')} → {chat['response'].get('response_language', 'Unknown')}")
                    else:
                        st.write(f"**Status:** {chat['response'].get('message', 'Error')}")
    
    with tab2:
        st.header("📁 Document Management")
        
        # File upload
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=['txt', 'md', 'pdf', 'docx', 'doc', 'csv', 'json', 'xml'],
            accept_multiple_files=True,
            help="Supported formats: Text, Markdown, PDF, Word, CSV, JSON, XML"
        )
        
        if uploaded_files:
            st.write(f"**Selected files:** {len(uploaded_files)}")
            
            # Display file info
            file_info = []
            for file in uploaded_files:
                file_info.append({
                    'Name': file.name,
                    'Size': f"{file.size / 1024:.1f} KB",
                    'Type': file.type or 'Unknown'
                })
            
            st.dataframe(pd.DataFrame(file_info))
            
            # Process documents
            if st.button("🔄 Process & Index Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    # Save uploaded files temporarily
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    
                    try:
                        for file in uploaded_files:
                            temp_path = os.path.join(temp_dir, file.name)
                            with open(temp_path, 'wb') as f:
                                f.write(file.getvalue())
                            file_paths.append(temp_path)
                        
                        # Process documents
                        result = st.session_state.rag_engine.process_and_index_documents(file_paths)
                        
                        if result['status'] == 'success':
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Successfully processed {result['successful_docs']} documents!")
                            
                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Documents", result['successful_docs'])
                            col2.metric("Chunks", result['total_chunks'])
                            col3.metric("Processing Time", f"{result['processing_time']:.2f}s")
                            col4.metric("Failed", result['failed_docs'])
                            
                            # Show processing stats
                            if 'processing_stats' in result:
                                stats = result['processing_stats']
                                st.subheader("📊 Processing Statistics")
                                
                                if 'language_distribution' in stats:
                                    st.write("**Language Distribution:**")
                                    lang_df = pd.DataFrame([
                                        {'Language': config.SUPPORTED_LANGUAGES.get(lang, lang), 'Count': count}
                                        for lang, count in stats['language_distribution'].items()
                                    ])
                                    st.bar_chart(lang_df.set_index('Language'))
                        
                        else:
                            st.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
                    
                    finally:
                        # Clean up temporary files
                        shutil.rmtree(temp_dir)
        
        # Document statistics
        if st.session_state.documents_loaded or st.session_state.sample_docs_added:
            st.subheader("📊 Document Statistics")
            
            try:
                db_info = st.session_state.rag_engine.vector_db.get_database_info()
                if 'collection_stats' in db_info:
                    stats = db_info['collection_stats']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Documents", stats.get('total_documents', 0))
                    col2.metric("Total Chunks", stats.get('total_chunks', 0))
                    col3.metric("Languages", len(stats.get('language_distribution', {})))
                    
                    # Language distribution chart
                    if 'language_distribution' in stats:
                        st.write("**Language Distribution:**")
                        lang_data = []
                        for lang, count in stats['language_distribution'].items():
                            lang_name = config.SUPPORTED_LANGUAGES.get(lang, lang)
                            lang_data.append({'Language': lang_name, 'Count': count})
                        
                        if lang_data:
                            lang_df = pd.DataFrame(lang_data)
                            st.bar_chart(lang_df.set_index('Language'))
                
            except Exception as e:
                st.warning(f"Could not retrieve database statistics: {e}")
    
    with tab3:
        st.header("📊 System Analytics")
        
        if st.session_state.rag_engine:
            # System status
            st.subheader("🔧 System Status")
            system_status = st.session_state.rag_engine.get_system_status()
            
            # Display system information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**RAG Engine:**")
                st.write(f"- Status: {system_status.get('rag_engine_status', 'Unknown')}")
                
                st.write("**Document Processor:**")
                doc_proc = system_status.get('document_processor', {})
                st.write(f"- Status: {doc_proc.get('status', 'Unknown')}")
                st.write(f"- Supported Formats: {', '.join(doc_proc.get('supported_formats', []))}")
            
            with col2:
                st.write("**Embedding Engine:**")
                emb_engine = system_status.get('embedding_engine', {})
                st.write(f"- Model: {emb_engine.get('model_name', 'Unknown')}")
                st.write(f"- Dimension: {emb_engine.get('embedding_dimension', 'Unknown')}")
                st.write(f"- Device: {emb_engine.get('device', 'Unknown')}")
                
                st.write("**Translation Service:**")
                trans_service = system_status.get('translation_service', {})
                st.write(f"- Status: {trans_service.get('status', 'Unknown')}")
                st.write(f"- Supported Languages: {trans_service.get('supported_languages', 'Unknown')}")
            
            # Performance metrics
            st.subheader("⚡ Performance Metrics")
            
            # Add sample performance data (in real system, this would come from actual usage)
            performance_data = {
                'Metric': ['Average Response Time', 'Success Rate', 'Language Coverage', 'Document Processing Speed'],
                'Value': ['2.3s', '87%', '12 languages', '15 docs/min'],
                'Status': ['✅ Good', '✅ Good', '✅ Excellent', '✅ Good']
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
    
    with tab4:
        st.header("🧪 System Testing")
        
        if st.session_state.rag_engine:
            # Test queries
            st.subheader("🔍 Test Queries")
            
            # Predefined test queries in multiple languages
            test_queries = [
                {"query": "What are the symptoms of diabetes?", "language": "en"},
                {"query": "¿Cuáles son los síntomas de la diabetes?", "language": "es"},
                {"query": "Quels sont les symptômes du diabète?", "language": "fr"},
                {"query": "Was sind die Symptome von Diabetes?", "language": "de"},
                {"query": "糖尿病の症状は何ですか？", "language": "ja"}
            ]
            
            # Test individual queries
            st.write("**Test Individual Queries:**")
            for i, test_case in enumerate(test_queries):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i+1}. {test_case['query']} ({test_case['language']})")
                with col2:
                    if st.button(f"Test {i+1}", key=f"test_{i}"):
                        with st.spinner(f"Testing query {i+1}..."):
                            result = st.session_state.rag_engine.query(
                                test_case['query'],
                                target_language=response_language
                            )
                            
                            if result['status'] == 'success':
                                st.success("✅ Success")
                                st.write(f"**Answer:** {result['answer'][:100]}...")
                            else:
                                st.error(f"❌ Failed: {result.get('message', 'Unknown error')}")
            
            # Batch testing
            st.subheader("📋 Batch Testing")
            if st.button("🧪 Run Batch Test", use_container_width=True):
                with st.spinner("Running batch test..."):
                    evaluation_result = st.session_state.rag_engine.evaluate_system(test_queries)
                    
                    if evaluation_result:
                        st.success("✅ Batch test completed!")
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Queries", evaluation_result.get('total_queries', 0))
                        col2.metric("Success Rate", f"{evaluation_result.get('success_rate', 0)*100:.1f}%")
                        col3.metric("Avg Response Time", f"{evaluation_result.get('average_response_time', 0):.2f}s")
                        col4.metric("Languages", len(evaluation_result.get('language_coverage', [])))
                        
                        # Language coverage
                        if 'language_coverage' in evaluation_result:
                            st.write("**Language Coverage:**")
                            for lang in evaluation_result['language_coverage']:
                                lang_name = config.SUPPORTED_LANGUAGES.get(lang, lang)
                                st.write(f"- {lang_name} ({lang})")
                    else:
                        st.error("❌ Batch test failed")

if __name__ == "__main__":
    main()
