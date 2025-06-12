from VectorIndexer import VectorIndexer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import SpacyEmbeddings
from langchain_core.documents import Document
import uuid
import faiss
import re
import unicodedata
import json
import spacy
import os

class FaissIndexer(VectorIndexer):
    def __init__(self, path, collection_name,spacy_model_name):
        self.path = path
        self.collection_name = collection_name
        self.index_path = os.path.join(path, f"{collection_name}.faiss")
        
        # Initialize SpaCy embedding model
        self.embedding = SpacyEmbeddings(model_name=spacy_model_name)
        self.nlp = spacy.load(spacy_model_name)
        self.vector_size = self.nlp.vocab.vectors_length
        self.dimension = self.nlp("test").vector.shape[0]
        print(f"dimension:{self.dimension}")
        self.vector_store = None
        self.doc_store = {}  # Stores all documents by ID
        self.id_to_docstore_id = {}  # Maps our ID to FAISS docstore ID
        self.initialized = False


        print("--- Initializing ---")
        self.initialize_state()
        #self._load_from_disk()
    
    def initialize_state(self):
        """Initialize the FAISS index with an empty state"""
        empty_docs = [Document(page_content="", metadata={})]

        # initialize vector stor
        self.vector_store = FAISS(
            embedding_function=self.embedding,
            index=faiss.IndexFlatL2(self.dimension),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        # Remove the dummy document
        self.doc_store = {}
        self.id_to_docstore_id = {}
        self.initialized = True

    def add_documents(self, documents : list[Document]):
        if not self.initialized:
            self.initialize_state()
        print(f"len of docs {len(documents)}")
        self.id_to_docstore_id = { f"{doc.metadata['id']}_{doc.metadata["chunk_index"]}": str(uuid.uuid4()) for doc in documents }
        ids = [val for key,val in self.id_to_docstore_id.items()]
        self.vector_store.add_documents(documents,ids=ids)
        self._save_to_disk()
    
    def update_documents(self, ids, documents, metadatas):
        # For simplicity, remove and re-add (FAISS doesn't support direct updates)
        """
        Update documents in the index
        
        Args:
            ids: List of document IDs to update
            documents: List of updated document contents
            metadatas: List of updated metadata dictionaries
        """
        if not self.initialized or not ids:
            return
        
        # First delete the existing documents
        self.delete_documents(ids)
        
        # Then add the updated versions
        self.add_documents(ids, documents, metadatas)
    
    def query(self, query_texts, n_results=5, where=None):
        """
        Query the index
        
        Args:
            query_texts: List of query strings
            n_results: Number of results to return per query
            where: Optional metadata filter (not implemented in this version)
            
        Returns:
            List of results for each query, where each result is a tuple of:
            (document_id, content, metadata, similarity_score)
        """
        if not self.initialized:
            return [[] for _ in query_texts]
        
        all_results = []
        for query in query_texts:
            # Perform similarity search
            docs = self.vector_store.similarity_search_with_score(
                query, 
                k=n_results
                # meta data filter goes here
            )
            
            # Process results
            query_results = []
            for doc, score in docs:
                doc_id = doc.metadata.get("document_id")
                if doc_id:
                    query_results.append((
                        doc_id,
                        doc.page_content,
                        doc.metadata,
                        float(1 - score)  # Convert distance to similarity
                    ))
            all_results.append(query_results)
        
        return all_results
    
    def get_all_documents(self):
        """
        Get all documents from the index
        
        Returns:
            List of tuples (document_id, content, metadata)
        """
        return [
            (doc_id, data["content"], data["metadata"])
            for doc_id, data in self.doc_store.items()
        ]

    def count(self):
        """Get the count of documents in the index"""
        return len(self.doc_store)
    
    def _load_from_disk(self):
        """ Loads the vector store if it exist on the local disk """
        if os.path.exists(self.index_path):
            print("--- Vector Store files found; LOADING ---")
            self.vector_store = FAISS.load_local(self.index_path,self.embedding)
            self.initialize_state = True
    
    def _save_to_disk(self):
        """ Saves the vector store to disk for persistance"""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
        else:
            print("--- Error: No vector store initialized ---")  

    def _delete_documents(self, ids):
        """
        Delete documents from the index (helper method)
        
        Args:
            ids: List of document IDs to delete
        """
        if not self.initialized:
            return
        
        # Get FAISS docstore IDs for deletion
        uuids_to_delete = []
        for doc_id in ids:
            if doc_id in self.id_to_docstore_id:
                uuids_to_delete.append(self.id_to_docstore_id[doc_id])
                # Remove from our stores
                del self.id_to_docstore_id[doc_id]
        
        # Perform deletion in FAISS
        if uuids_to_delete:
            self.vector_store.delete(uuids_to_delete)


def clean_text(text, lower=False, remove_special_chars=False, max_consecutive_whitespace=3):
    """
    Cleans technical text while preserving critical details like:
    - Code snippets
    - IP addresses
    - File paths
    - Command examples
    
    Args:
        text (str): Input text to clean
        lower (bool): Whether to lowercase text (default=False)
        remove_special_chars (bool): Remove non-alphanumeric chars (default=False)
        max_consecutive_whitespace (int): Max allowed consecutive whitespaces
    
    Returns:
        str: Cleaned technical text
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Normalize Unicode (e.g., convert curly quotes to straight)
    text = unicodedata.normalize('NFKC', text)
    
    # Preserve technical patterns before cleaning
    preserved_patterns = {
        r'(?:\[\d+\])': ' ',       # Remove [123] style indexes
        r'`(.+?)`': r'\1',          # Keep code inside backticks but remove ticks
        r'(?<!\\)\\(?!\\)': ' ',   # Remove isolated backslashes
    }
    
    for pattern, replacement in preserved_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove non-printable characters except tabs/newlines
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Handle whitespace normalization
    text = re.sub(r'\t', ' ', text)                      # Tabs to spaces
    text = re.sub(r'\r\n', '\n', text)                   # Normalize line endings
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)         # Single newlines to space
    text = re.sub(r' {2,}', ' ', text)                   # Multiple spaces to one
    text = re.sub(r'\n{2,}', '\n\n', text)               # Limit consecutive newlines
    
    # Optional: Remove special characters (preserves tech patterns)
    if remove_special_chars:
        # Keep: alphanumeric, basic punctuation, tech symbols (@.:/_~-)
        text = re.sub(r'[^\w\s@.:/~_-]', '', text)
    
    # Optional case normalization
    if lower:
        # Preserve case in known acronyms (IP, CPU, HTTP)
        acronyms = {'ip', 'cpu', 'gpu', 'ram', 'http', 'https', 'ssh', 'tcp', 'udp'}
        words = []
        for word in text.split():
            if word.lower() in acronyms and word.isupper():
                words.append(word.lower())
            else:
                words.append(word.lower() if lower else word)
        text = ' '.join(words)
    
    # Final cleanup
    text = text.strip()
    
    # Ensure max consecutive whitespace
    if max_consecutive_whitespace:
        text = re.sub(
            r'\s{{{},}}'.format(max_consecutive_whitespace+1), 
            ' ' * max_consecutive_whitespace, 
            text
        )
    
    return text

def preprocess(text,vector_store):
  nlp = vector_store.nlp
  doc = nlp(str(text))
  stop_words = nlp.Defaults.stop_words
  preprocessed_text = []
  for token in doc:
    if token.is_punct or token.like_num or token in stop_words or token.is_space:
      continue
    preprocessed_text.append(token.lemma_.lower().strip())
  return ' '.join(preprocessed_text)

def retrieve_full_entry(vector_store, doc_store, query: str, k: int = 5):
    """Search vector store and return full JSON entries in relevance order"""
    # 1. Perform similarity search with scores
    results = vector_store.similarity_search_with_score(query, k=k)
    # 2. Track document relevance using best chunk score
    chunk_groups = {}
    
    for doc, score in results:
        print(chunk_groups)
        doc_id = doc.metadata["id"]
        print(f"evaluating {doc_id}")
        # Initialize document entry
        if doc_id not in chunk_groups:
            chunk_groups[doc_id] = {"min_score": float('inf')}
        print(f"current score is {chunk_groups[doc_id]["min_score"]}")
        # Track best (lowest) score for document
        if score < chunk_groups[doc_id]["min_score"]:
            print(f"score updated: from {chunk_groups[doc_id]["min_score"]} to {score}")
            chunk_groups[doc_id]["min_score"] = score

        #chunk_groups[doc_id]["chunks"].append(doc)

    # 3. Sort documents by relevance (best score first)
    sorted_doc_ids = sorted(
        chunk_groups.keys(),
        key=lambda doc_id: chunk_groups[doc_id]["min_score"]
    )
    # 4. Assemble final results in relevance order
    full_results = []
    for doc_id in sorted_doc_ids:
        # data = chunk_groups[doc_id]
        # data["chunks"].sort(key=lambda x: x.metadata["chunk_index"])
        # assembled_content = " ".join(chunk.page_content for chunk in data["chunks"])
        
        full_results.append({
            "id": doc_id,
            "full_entry": doc_store[doc_id],
            "min_score": chunk_groups[doc_id]["min_score"]  # Optional: for debugging
        })
    
    return full_results

def testing_function():
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
    with open("src/data/data_center_tickets_call_1.json",'r') as f:
        json_data = json.load(f)
    all_docs = []
    doc_store = {}  # To store full JSON entries for retrieval
    fais_indexer = FaissIndexer("src/data/","faiss_index.fais","en_core_web_md")
    vector_store = fais_indexer.vector_store

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,          # Character-based splitting
    chunk_overlap=15,
    separators=["\n\n", "\n", ". ", " ", ""]
    )
    

    # text_splitter = TokenTextSplitter(
    # chunk_size=200,           # Optimal for most sentence transformers
    # chunk_overlap=100,
    # encoding_name="cl100k_base"  # For OpenAI/compatible models
    # )

    for entry in json_data[:5]:
        print(entry["ticket_number"])
        doc_store[entry["ticket_number"]] = entry  # Store full entry
        cleaned_text = clean_text(entry["issue_description"])
        cleaned_text = preprocess(cleaned_text,fais_indexer)
        # Only chunk if contsent exceeds threshold (300 chars)
        if len(entry["issue_description"]) > 100:
            chunks = text_splitter.split_text(cleaned_text)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "id": entry["ticket_number"],
                    "network": entry["network"],
                    "responsibility": entry["responsibility"],
                    "main_ticket": entry["main_office_ticket"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_length": len(entry["issue_description"])
                }
                all_docs.append(Document(page_content=chunk, metadata=metadata))
            print(f"processed entry {entry["ticket_number"]} in {len(chunks)} chunks")
        else:
            metadata = {
                "id": entry["ticket_number"],
                "network": entry["network"],
                "responsibility": entry["responsibility"],
                "main_ticket": entry["main_office_ticket"],
                "chunk_index": 0,
                "total_chunks": 1
            }
            all_docs.append(Document(page_content=cleaned_text, metadata=metadata))
    ## adding docsvia the main method
    #vector_store.add_documents(all_docs)
    fais_indexer.add_documents(all_docs)

    # fais_indexer = None

    # fais_indexer = FaissIndexer("src/data/","faiss_index.fais","en_core_web_md")

    query = "Servers with high cpu usage"
    query = "linux server issue"
    results = retrieve_full_entry(vector_store,doc_store,query)

    print("\nTop results for query:", query)
    for i, res in enumerate(results):
        print(f"\nResult #{i+1}:")
        print(f"id: {res['full_entry']['ticket_number']}")
        print(f"network: {res['full_entry']['network']}")
        print(f"responsibility: {res['full_entry']['responsibility']}")
        print(f"main_ticket: {res['full_entry']['main_office_ticket']}")
        print(f"content: {res['full_entry']["issue_description"][:150]}...")
        print(f"With chunk with min score: {res["min_score"]}")

if __name__ == '__main__':
    testing_function() 