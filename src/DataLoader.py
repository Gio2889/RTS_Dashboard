import pandas as pd
import requests
import threading
import time
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import json
import openai
import os
from datetime import datetime

# Set OpenAI API key (store securely in environment variables)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

class DataLoader:
    def __init__(self, api_url, status_column, closed_status, 
                 index_condition, index_fields, interval=300):
        self.api_url = api_url
        self.status_column = status_column
        self.closed_status = closed_status
        self.index_condition = index_condition
        self.index_fields = index_fields
        self.interval = interval
        
        # Initialize DataFrame
        self.data = pd.DataFrame()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_func = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="tickets",
            embedding_function=self.embedding_func
        )
        
        # Track document states
        self.indexed_ids = set()
        self.last_hashes = {}
        self._init_chromadb_state()
        
        # Initial data load
        self._fetch_data()
        
        # Start background thread
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _init_chromadb_state(self):
        """Initialize ChromaDB state tracking"""
        existing = self.collection.get(include=["metadatas", "documents", "ids"])
        for doc_id, metadata, document in zip(existing['ids'], existing['metadatas'], existing['documents']):
            self.indexed_ids.add(doc_id)
            doc_hash = hashlib.sha256(document.encode()).hexdigest()
            self.last_hashes[doc_id] = doc_hash

    def _fetch_data(self):
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            new_data = pd.DataFrame(response.json())
            
            if new_data.empty:
                return
                
            with self.lock:
                if self.data.empty:
                    self.data = new_data
                    self._process_initial_indexing()
                    return
                
                self._process_data_update(new_data)
                
        except Exception as e:
            st.error(f"API Error: {e}")

    def _process_initial_indexing(self):
        """Initial ChromaDB indexing"""
        to_index = self.data[self.data[self.index_condition]]
        for _, row in to_index.iterrows():
            doc_id = str(row['id'])
            content = self._create_document_content(row)
            metadata = self._create_document_metadata(row)
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            self.indexed_ids.add(doc_id)
            self.last_hashes[doc_id] = hashlib.sha256(content.encode()).hexdigest()

    def _process_data_update(self, new_data):
        """Update both DataFrame and ChromaDB with new data"""
        updated_ids = set()
        for _, new_row in new_data.iterrows():
            row_id = str(new_row['id'])
            
            # Find existing row
            existing_idx = self.data.index[self.data['id'] == new_row['id']]
            
            if not existing_idx.empty:
                # Update existing row
                idx = existing_idx[0]
                if not self._row_equal(self.data.loc[idx], new_row):
                    self.data.loc[idx] = new_row
                    updated_ids.add(row_id)
            else:
                # Add new row
                self.data = pd.concat([self.data, new_row.to_frame().T], ignore_index=True)
                updated_ids.add(row_id)
        
        # Process ChromaDB updates
        self._update_chromadb(updated_ids)

    def _row_equal(self, row1, row2):
        """Check if two rows are identical"""
        return row1.to_json() == row2.to_json()

    def _update_chromadb(self, updated_ids):
        """Update ChromaDB for modified/new rows"""
        for doc_id in updated_ids:
            # Get the row from DataFrame
            row = self.data[self.data['id'].astype(str) == doc_id].iloc[0]
            
            # Skip closed tickets
            if row[self.status_column] == self.closed_status:
                continue
                
            # Create document content
            content = self._create_document_content(row)
            current_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Skip if content hasn't changed
            if doc_id in self.last_hashes and self.last_hashes[doc_id] == current_hash:
                continue
                
            metadata = self._create_document_metadata(row)
            
            if doc_id in self.indexed_ids:
                # Update existing document
                self.collection.update(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[metadata]
                )
            elif row[self.index_condition]:
                # Add new document if condition met
                self.collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                self.indexed_ids.add(doc_id)
            
            # Update hash
            self.last_hashes[doc_id] = current_hash

    def _create_document_content(self, row):
        """Create document content from row"""
        return "\n".join([f"{field}: {row[field]}" for field in self.index_fields])

    def _create_document_metadata(self, row):
        """Create document metadata from row"""
        metadata = {field: str(row[field]) for field in self.index_fields}
        metadata['status'] = row[self.status_column]
        metadata['id'] = str(row['id'])
        return metadata

    def _update_loop(self):
        """Background update process"""
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self._fetch_data()
            
    def search_tickets(self, query, n_results=5, filter_status=None):
        """Search tickets with optional status filter"""
        where = {}
        if filter_status:
            where = {"status": filter_status}
            
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
    
    def get_data(self):
        """Get current data snapshot"""
        with self.lock:
            return self.data.copy()
    
    def stop(self):
        """Stop background updates"""
        self.stop_event.set()
        self.thread.join()

class ChatAssistant:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.system_prompt = (
            "You are a helpful support assistant. Use the provided ticket information to answer questions. "
            "When asked about solutions, provide the most relevant solutions from similar past tickets. "
            "Be concise and factual. If you don't know, say so."
        )
    
    def generate_response(self, user_query, n_retrieve=5, history=None):
        """Generate response using RAG with ticket context"""
        # Retrieve relevant tickets
        results = self.data_loader.search_tickets(user_query, n_results=n_retrieve)
        
        # Format context for LLM
        context = "Relevant Tickets:\n"
        if results['ids'][0]:
            for i, (doc_id, doc, meta) in enumerate(zip(results['ids'][0], results['documents'][0], results['metadatas'][0])):
                context += f"\n### Ticket {meta['id']} ({meta['status']})\n{doc}\n"
        else:
            context += "No relevant tickets found."
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history if history else [],
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": context}
        ]
        
        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Streamlit App
@st.cache_resource
def init_data_loader():
    return DataLoader(
        api_url="https://api.example.com/tickets",
        status_column="status",
        closed_status="closed",
        index_condition="ready_for_llm",
        index_fields=["id", "title", "description", "category", "priority", "solution"],
        interval=300
    )

def main():
    st.set_page_config(layout="wide", page_title="Ticket Management System")
    
    # Initialize components
    loader = init_data_loader()
    assistant = ChatAssistant(loader)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Create tabs
    tab1, tab2 = st.tabs(["Ticket Management", "LLM Support Assistant"])
    
    with tab1:
        st.title("Ticket Management Dashboard")
        data = loader.get_data()
        
        if not data.empty:
            # Calculate metrics
            open_tickets = data[data['status'] != 'closed']
            closed_tickets = data[data['status'] == 'closed']
            avg_resolution = pd.to_timedelta(data[data['status'] == 'closed']['resolution_time']).mean()
            
            # Display KPIs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tickets", len(data))
            col2.metric("Open Tickets", len(open_tickets))
            col3.metric("Closed Tickets", len(closed_tickets))
            col4.metric("Avg Resolution", str(avg_resolution).split('.')[0] if not pd.isna(avg_resolution) else "N/A")
            
            # Display data table
            st.subheader("Ticket Data")
            st.dataframe(data.sort_values(by='created_at', ascending=False))
            
            # Last update indicator
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("No ticket data available")
            
        # Manual refresh button
        if st.button("Force Refresh Data"):
            loader._fetch_data()
            st.rerun()
    
    with tab2:
        st.title("Support Assistant")
        st.info("Ask questions about tickets or search for solutions to similar issues")
        
        # Chat configuration
        col1, col2 = st.columns(2)
        n_results = col1.slider("Number of tickets to retrieve", 3, 10, 5)
        rag_toggle = col2.toggle("Enable RAG Retrieval", True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("Sources"):
                        st.json(message["sources"])
        
        # User input
        if user_query := st.chat_input("Ask about tickets or solutions..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching tickets..."):
                    # Retrieve relevant tickets
                    if rag_toggle:
                        results = loader.search_tickets(user_query, n_results=n_results)
                        sources = {
                            "ids": results['ids'][0],
                            "documents": results['documents'][0],
                            "metadatas": results['metadatas'][0]
                        }
                    else:
                        sources = {"message": "RAG retrieval disabled"}
                    
                    # Generate LLM response
                    response = assistant.generate_response(
                        user_query,
                        n_retrieve=n_results,
                        history=st.session_state.chat_history
                    )
                
                # Display assistant response
                st.markdown(response)
                
                # Show sources if RAG enabled
                if rag_toggle and results['ids'][0]:
                    with st.expander("Retrieved Tickets"):
                        for i, (doc_id, doc, meta) in enumerate(zip(sources['ids'], sources['documents'], sources['metadatas'])):
                            st.subheader(f"Ticket {meta['id']} ({meta['status']})")
                            st.caption(f"Similarity: {results['distances'][0][i]:.3f}")
                            st.write(doc)
                            st.divider()
                
                # Add response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })

if __name__ == "__main__":
    main()