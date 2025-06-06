from VectorIndexer import VectorIndexer
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
        
        # Initialize SpaCy embedding model
        self.embedding = SpacyEmbeddings(model_name=spacy_model_name)
        self.nlp = spacy.load(spacy_model_name)
        self.vector_size = self.nlp.vocab.vectors_length
        self.dimension = self.nlp("test").vector.shape[0]
        
        self.vector_store = None
        self.doc_store = {}  # Stores all documents by ID
        self.id_to_docstore_id = {}  # Maps our ID to FAISS docstore ID
        self.initialized = False


        print("--- Initializing ---")
        self.initialize_state()
        self._load_from_disk()
    
    def initialize_state(self):
        """Initialize the FAISS index with an empty state"""
        empty_docs = [Document(page_content="", metadata={})]
        self.vector_store = FAISS.from_documents(

            documents=empty_docs,
            embedding=self.embedding
        )
        # Remove the dummy document
        self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
        self.doc_store = {}
        self.id_to_docstore_id = {}
        self.initialized = True

    def add_documents(self, documents : list[Document]):
         if not self.initialized:
             self.initialize_state()
         self.vector_store.add_documents(documents)
         self._save_to_disk()


    # def add_non_proc_documents(self, ids : list , documents : list , metadatas: list):
    #     """
    #     Add documents to the index
        
    #     Args:
    #         ids: List of unique document IDs
    #         documents: List of document contents 
    #         metadatas: List of metadata dictionaries
    #     """
    #     if not self.initialized:
    #         self.initialize_state()
        
    #     # Create Document objects
    #     docs = []
    #     for doc_id, content, meta in zip(ids, documents, metadatas):
    #         # Add our ID to metadata for retrieval
    #         meta["document_id"] = doc_id
    #         docs.append(Document(page_content=content, metadata=meta))

    #         # vector store should always exist no check needed here
    #         self.vector_store.add_documents(docs)
        
    #     # Update our ID mapping
    #     for doc_id, doc in zip(ids, docs):
    #         # Find the FAISS docstore ID for our document
    #         # This assumes documents are added in the same order
    #         idx = self.vector_store.docstore._dict
    #         for uuid, stored_doc in idx.items():
    #             if stored_doc.metadata.get("document_id") == doc_id:
    #                 self.id_to_docstore_id[doc_id] = uuid
    #                 break
    
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


def retrieve_full_entry(vector_store,doc_store,query: str, k: int = 5):
    """Search vector store and return full JSON entries"""
    # 1. Perform similarity search
    results = vector_store.similarity_search(query, k=k)
    
    # 2. Group results by document ID
    grouped_results = {}
    for doc in results:
        doc_id = doc.metadata["id"]
        if doc_id not in grouped_results:
            grouped_results[doc_id] = {
                "chunks": [],
                "metadata": {k:v for k,v in doc.metadata.items() if k != "chunk_index"}
            }
        grouped_results[doc_id]["chunks"].append(doc)
    
    # 3. Retrieve full entries and assemble context
    full_results = []
    for doc_id, data in grouped_results.items():
        # Sort chunks by index
        data["chunks"].sort(key=lambda x: x.metadata["chunk_index"])
        assembled_content = " ".join([chunk.page_content for chunk in data["chunks"]])
        
        # Get full JSON entry
        full_entry = doc_store[doc_id]
        
        full_results.append({
            "id": doc_id,
            "full_entry": full_entry,
            "assembled_content": assembled_content,
            "retrieved_chunks": len(data["chunks"])
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
    all_docs = []
    doc_store = {}  # To store full JSON entries for retrieval

    fais_indexer = FaissIndexer("../data/","faiss_index.fais","en_core_web_md")
    vector_store = fais_indexer.vector_store

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # Character-based splitting
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

    for entry in json_data:
        doc_store[entry["id"]] = entry  # Store full entry
        
        # Only chunk if content exceeds threshold (300 chars)
        if len(entry["content"]) > 300:
            chunks = text_splitter.split_text(entry["content"])
            for i, chunk in enumerate(chunks):
                metadata = {
                    "id": entry["id"],
                    "title": entry["title"],
                    "author": entry["author"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_length": len(entry["content"])
                }
                all_docs.append(Document(page_content=chunk, metadata=metadata))
            print(f"processed entry {entry["id"]} in {len(chunks)} chunks")
        else:
            metadata = {
                "id": entry["id"],
                "title": entry["title"],
                "author": entry["author"],
                "chunk_index": 0,
                "total_chunks": 1
            }
            all_docs.append(Document(page_content=entry["content"], metadata=metadata))

    ## adding docsvia the main method
    #vector_store.add_documents(all_docs)

    fais_indexer.add_documents(all_docs)

    fais_indexer = None

    fais_indexer = FaissIndexer("../data/","faiss_index.fais","en_core_web_md")

    query = "Alzheimer's disease treatment"
    results = retrieve_full_entry(vector_store,doc_store,query)

    print("\nTop results for query:", query)
    for i, res in enumerate(results):
        print(f"\nResult #{i+1}:")
        print(f"Title: {res['full_entry']['title']}")
        print(f"Author: {res['full_entry']['author']}")
        print(f"Content Preview: {res['assembled_content'][:150]}...")
        print(f"Retrieved {res['retrieved_chunks']} chunks from {res['full_entry']['id']}")

if __name__ == '__main__':
    testing_function()