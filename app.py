import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

# Page configuration (must be first Streamlit command)
try:
    st.set_page_config(
        page_title="JavaScript Course Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    # If running outside Streamlit, this will fail - that's expected
    pass

# Custom CSS styling function
def load_css():
    """Load custom CSS styling"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .response-container {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin-top: 1rem;
        }
        .video-info {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .timestamp {
            color: #1f77b4;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the embeddings dataframe (cached for performance)"""
    try:
        df = joblib.load("chunks_embeddings.joblib")
        return df
    except FileNotFoundError:
        st.error("❌ Error: chunks_embeddings.joblib file not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def create_embedding(text_list):
    """Create embeddings using Ollama API"""
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": text_list
            },
            timeout=30
        )
        r.raise_for_status()
        return r.json()["embeddings"]
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error connecting to Ollama API: {str(e)}")
        st.info("💡 Make sure Ollama is running with: `ollama serve`")
        return None

def inference(prompt, stream=False, status_container=None):
    """Generate response using Ollama LLM with optional streaming"""
    try:
        if stream:
            # Streaming mode - provides real-time feedback
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": True
                },
                timeout=600,  # 10 minutes for streaming
                stream=True
            )
            r.raise_for_status()
            
            full_response = ""
            if status_container:
                progress_text = status_container.empty()
            
            for line in r.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if "response" in json_response:
                            token = json_response["response"]
                            full_response += token
                            if status_container and progress_text:
                                # Update progress every 50 characters to reduce UI updates
                                if len(full_response) % 50 == 0 or len(full_response) < 50:
                                    progress_text.info(f"🤖 Generating... ({len(full_response)} characters so far)")
                        if json_response.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            if status_container:
                progress_text.empty()
            
            return full_response
        else:
            # Non-streaming mode (original)
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=600  # Increased to 10 minutes (600 seconds)
            )
            r.raise_for_status()
            response = r.json()
            return response.get("response", "No response generated")
    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timed out. The model is taking too long to respond."
        st.error(error_msg)
        st.info("""
        **Possible solutions:**
        - The prompt might be too large. Try asking a more specific question.
        - Ollama might be overloaded. Try again in a moment.
        - Consider using a faster model or reducing the number of chunks.
        """)
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Ollama API")
        st.info("💡 Make sure Ollama is running with: `ollama serve`")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error connecting to Ollama API: {str(e)}")
        st.info("💡 Make sure Ollama is running with: `ollama serve`")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def main():
    # Load custom CSS
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🎓 JavaScript Course Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your JavaScript course and get answers with video references!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Number of relevant chunks to retrieve", 3, 10, 5)
        use_streaming = st.checkbox("Use streaming mode (recommended)", value=True, 
                                     help="Shows real-time progress while generating response")
        st.markdown("---")
        
        # Chat history
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.header("💭 Chat History")
            for i, item in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
                with st.expander(f"Q: {item['query'][:50]}..."):
                    st.markdown(item['response'][:200] + "...")
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
            st.markdown("---")
        
        st.header("ℹ️ About")
        st.info("""
        This RAG-powered assistant helps you find answers 
        from your JavaScript course videos.
        
        **How it works:**
        1. Ask a question
        2. System finds relevant video chunks
        3. AI generates a helpful response
        4. Get video references with timestamps
        """)
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Status check
        st.markdown("---")
        st.header("🔌 Status")
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                st.success("✅ Ollama is running")
            else:
                st.error("❌ Ollama connection issue")
        except:
            st.error("❌ Ollama not running")
            st.caption("Run: `ollama serve`")
    
    # Load data
    with st.spinner("📚 Loading course data..."):
        df = load_data()
    
    st.success(f"✅ Loaded {len(df)} video chunks!")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main content area
    st.markdown("### 💬 Ask a Question")
    col1, col2 = st.columns([4, 1])
    
    
    with col1:
        query = st.text_input(
            "Ask your question",
            placeholder="e.g., What is a closure in JavaScript?",
            key="query_input",
            label_visibility="collapsed"
        )
    
    
    with col2:
        # st.write("")  # Spacing
        submit_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and query:
        if not query.strip():
            st.warning("⚠️ Please enter a question!")
        else:
            with st.spinner("🔍 Searching relevant content..."):
                # Get embedding for query
                question_embedding = create_embedding([query])
                
                if question_embedding is None:
                    st.stop()
                
                question_embedding = question_embedding[0]
                
                # Compute similarities
                emb_matrix = np.vstack(df["embedding"].values)
                similarities = cosine_similarity(emb_matrix, [question_embedding]).flatten()
                
                # Get top-k most similar chunks
                top_indices = similarities.argsort()[::-1][:top_k]
                result_df = df.iloc[top_indices].copy()
                result_df = result_df.reset_index(drop=True)
                
                # Display relevant chunks
                with st.expander(f"📋 Top {top_k} Relevant Video Segments", expanded=False):
                    for idx, row in result_df.iterrows():
                        similarity_score = similarities[top_indices[idx]]
                        st.markdown(f"""
                        <div class="video-info">
                            <strong>Episode {row['number']}: {row['title']}</strong><br>
                            <span class="timestamp">⏱️ {format_timestamp(row['start'])} - {format_timestamp(row['end'])}</span>
                            <br>Relevance: {similarity_score:.2%}
                            <br><br>{row['text'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)
            
            # Generate response
            prompt = f'''I am teaching Javascript in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{result_df[["title", "number", "start", "end", "text"]].to_json()}
---------------------------------
"{query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
            
            if use_streaming:
                # Streaming mode with progress indicator
                status_placeholder = st.empty()
                with status_placeholder.container():
                    st.info("🤖 Generating response... This may take 1-3 minutes for complex queries.")
                
                response = inference(prompt, stream=True, status_container=status_placeholder)
                status_placeholder.empty()
            else:
                # Non-streaming mode
                with st.spinner("🤖 Generating response... This may take 1-3 minutes. Please wait..."):
                    response = inference(prompt, stream=False)
            
            # Display response (common for both modes)
            if response:
                # Save to chat history
                st.session_state.chat_history.append({
                    'query': query,
                    'response': response,
                    'videos': result_df[['number', 'title', 'start', 'end']].to_dict('records')
                })
                
                # Display response
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown("### 💡 Answer")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display quick video references
                st.markdown("### 📹 Quick Video References")
                cols = st.columns(min(3, len(result_df)))
                for idx, (_, row) in enumerate(result_df.iterrows()):
                    with cols[idx % len(cols)]:
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; height: 120px;">
                            <strong>Episode {row['number']}</strong><br>
                            <small>{row['title'][:50]}...</small><br>
                            <span style="color: #1f77b4; font-size: 1.1em;">⏱️ {format_timestamp(row['start'])}</span>
                        </div>
                        """, unsafe_allow_html=True)
    
    elif submit_button and not query:
        st.warning("⚠️ Please enter a question before searching!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Powered by RAG (Retrieval-Augmented Generation) | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Streamlit apps must be run with 'streamlit run app.py', not 'python app.py'
    # This check provides a helpful error message if run incorrectly
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is None:
            # Running without Streamlit context
            raise RuntimeError()
    except (ImportError, RuntimeError, AttributeError):
        import sys
        print("\n" + "="*70)
        print("❌ ERROR: This is a Streamlit app!")
        print("="*70)
        print("\n📝 Please run it with the correct command:\n")
        print("   streamlit run app.py")
        print("\n💡 Windows users can also use: start_ui.bat")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    main()
