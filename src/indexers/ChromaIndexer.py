# ChromaDB implementation
class ChromaIndexer(VectorIndexer):
    def __init__(self, path, collection_name):
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_func = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func
        )
    
    def initialize_state(self):
        existing = self.collection.get(include=["metadatas", "documents", "ids"])
        state = {}
        for doc_id, metadata, document in zip(existing['ids'], existing['metadatas'], existing['documents']):
            state[doc_id] = {
                'hash': hashlib.sha256(document.encode()).hexdigest(),
                'metadata': metadata,
                'document': document
            }
        return state
    
    def add_documents(self, ids, documents, metadatas):
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def update_documents(self, ids, documents, metadatas):
        self.collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    def query(self, query_texts, n_results, where=None):
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )
    
    def get_all_documents(self):
        return self.collection.get()
    
    def count(self):
        return self.collection.count()