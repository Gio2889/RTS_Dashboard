from src.utils.VectorIndexer import VectorIndexer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SpacyEmbeddings
from langchain_core.documents import Document
import json
import spacy
import os

class FaissIndexer(VectorIndexer):
    def __init__(self, path, collection_name,spacy_model_name):
        self.path = path
        self.collection_name = collection_name
        self.index_path = os.path.join(path, f"{collection_name}.faiss")
        
        # Initialize SpaCy embeddings
        self.embedding = SpacyEmbeddings(model_name=spacy_model_name)
        
        # Load the SpaCy model to get vector dimensions
        self.nlp = spacy.load(spacy_model_name)
        self.vector_size = self.nlp.vocab.vectors_length
        
        # Initialize SpaCy embedding model
        self.nlp = self._load_spacy_model()
        self.dimension = self.nlp("test").vector.shape[0]
        
        self.vector_store = None
        self.doc_store = {}  # Stores all documents by ID
        self.id_to_docstore_id = {}  # Maps our ID to FAISS docstore ID
        self.initialized = False
        
        # Load existing data
        self._load_from_disk()
    
    def _load_spacy_model(self, model_name="en_core_web_md"):
        try:
            return spacy.load(model_name)
        except OSError:
            import spacy.cli
            spacy.cli.download(model_name)
            return spacy.load(model_name)
    
    def _load_from_disk(self):
        if os.path.exists(self.index_path):
            self.vector_store = self.vector_store.load_local()
        
    
    def _save_to_disk(self):
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
        else:
            print("--- Error: No vector store initialized ---")
    
    def initialize_state(self):
        """ keep the hash of the content"""
        state = {}
        for doc_id, data in self.metadata.items():
            state[doc_id] = {
                'hash': hashlib.sha256(data['document'].encode()).hexdigest(),
                'metadata': data['metadata'],
                'document': data['document']
            }
        return state
    
    def add_documents(self, ids, documents, metadatas):
        # Generate embeddings
        embeddings = self._generate_embeddings(documents)
        
        # Convert to numpy array
        emb_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(emb_array)
        
        # Get starting index for new vectors
        start_idx = self.index.ntotal - len(ids)
        
        # Store metadata and mappings
        for i, doc_id in enumerate(ids):
            idx = start_idx + i
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
            self.metadata[doc_id] = {
                'document': documents[i],
                'metadata': metadatas[i]
            }
        
        self._save_to_disk()
    
    def update_documents(self, ids, documents, metadatas):
        # For simplicity, remove and re-add (FAISS doesn't support direct updates)
        self.remove_documents(ids)
        self.add_documents(ids, documents, metadatas)
    
    def remove_documents(self, ids):
        # This is a simplified implementation - in production, consider a more efficient method
        keep_indices = [i for i in range(self.index.ntotal) if self.index_to_id.get(i) not in ids]
        
        # Create a new index
        new_index = faiss.IndexFlatL2(self.dimension)
        
        # Add only the vectors we're keeping
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        keep_vectors = all_vectors[keep_indices]
        
        if keep_vectors.shape[0] > 0:
            new_index.add(keep_vectors)
        
        # Update metadata and mappings
        for doc_id in ids:
            if doc_id in self.metadata:
                del self.metadata[doc_id]
            if doc_id in self.id_to_index:
                del self.id_to_index[doc_id]
        
        # Reset mappings
        self.index = new_index
        self.index_to_id = {}
        self.id_to_index = {}
        
        # Rebuild mappings from metadata
        for doc_id in self.metadata.keys():
            # In a real implementation, you'd need to re-index
            pass
        
        self._save_to_disk()
    
    def _generate_embeddings(self, texts):
        return [doc.vector.tolist() for doc in self.nlp.pipe(texts)]
    
    def query(self, query_texts, n_results, where=None):
        # Generate query embeddings
        query_embeddings = self._generate_embeddings(query_texts)
        query_array = np.array(query_embeddings).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_array, n_results)
        
        
        results = {
            'ids': [],
            'distances': [],
            'metadatas': [],
            'documents': []
        }
        
        for i in range(len(query_texts)):
            ids_list = []
            distances_list = []
            metadatas_list = []
            documents_list = []
            
            for j, idx in enumerate(indices[i]):
                if idx < 0:
                    continue
                doc_id = self.index_to_id.get(idx)
                if doc_id and doc_id in self.metadata:
                    ids_list.append(doc_id)
                    distances_list.append(float(distances[i][j]))
                    metadatas_list.append(self.metadata[doc_id]['metadata'])
                    documents_list.append(self.metadata[doc_id]['document'])
            
            results['ids'].append(ids_list)
            results['distances'].append(distances_list)
            results['metadatas'].append(metadatas_list)
            results['documents'].append(documents_list)
        
        # Apply filtering if needed
        if where:
            filtered_results = {'ids': [], 'distances': [], 'metadatas': [], 'documents': []}
            for i in range(len(results['ids'])):
                filtered_ids = []
                filtered_distances = []
                filtered_metadatas = []
                filtered_documents = []
                
                for j, doc_id in enumerate(results['ids'][i]):
                    metadata = results['metadatas'][i][j]
                    match = True
                    for key, value in where.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_ids.append(doc_id)
                        filtered_distances.append(results['distances'][i][j])
                        filtered_metadatas.append(metadata)
                        filtered_documents.append(results['documents'][i][j])
                
                filtered_results['ids'].append(filtered_ids)
                filtered_results['distances'].append(filtered_distances)
                filtered_results['metadatas'].append(filtered_metadatas)
                filtered_results['documents'].append(filtered_documents)
            
            return filtered_results
        
        return results
    
    def get_all_documents(self):
        # Return in Chroma-like format
        return {
            'ids': list(self.metadata.keys()),
            'metadatas': [data['metadata'] for data in self.metadata.values()],
            'documents': [data['document'] for data in self.metadata.values()]
        }
    
    def count(self):
        return len(self.metadata)
    

if __name__ == '__main__':
    json_data = [
    {
        "id": "001",
        "title": "Tech Breakthrough",
        "author": "Jane Smith",
        "content": "Scientists have developed a new quantum processor that operates at room temperature."
    },
    {
        "id": "002",
        "title": "Climate Agreement",
        "author": "John Doe",
        "content": "Global leaders reached a historic agreement to reduce carbon emissions by 50% before 2030. The deal includes provisions for developing nations and establishes a carbon credit trading system."
    },
    {
        "id": "003",
        "title": "Medical Discovery",
        "author": "Dr. Alice Chen",
        "content": ("Researchers have discovered a new approach to treating Alzheimer's disease. " * 15)  # Long text
    },
    {
        "id": "004",
        "title": "Economic Forecast",
        "author": "Robert Johnson",
        "content": "The Federal Reserve announced new interest rate policies amid rising inflation concerns."
    },
    {
        "id": "005",
        "title": "Space Exploration",
        "author": "Sarah Williams",
        "content": "NASA's new telescope has captured unprecedented images of distant galaxies."
    }
]