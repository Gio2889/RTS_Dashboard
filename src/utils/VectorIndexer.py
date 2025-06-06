from abc import ABC, abstractmethod
class VectorIndexer(ABC):
    @abstractmethod
    def initialize_state(self):
        """Initialize the indexer state"""
        pass
    
    @abstractmethod
    def add_documents(self, ids, documents, metadatas):
        """Add documents to the index"""
        pass
    
    @abstractmethod
    def update_documents(self, ids, documents, metadatas):
        """Update documents in the index"""
        pass
    
    @abstractmethod
    def query(self, query_texts, n_results, where=None):
        """Query the index"""
        pass
    
    @abstractmethod
    def get_all_documents(self):
        """Get all documents from the index"""
        pass
    
    @abstractmethod
    def count(self):
        """Get the count of documents in the index"""
        pass