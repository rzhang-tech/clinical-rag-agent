from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

try:
    from fastembed import TextEmbedding
    _reranker_available = True
except ImportError:
    _reranker_available = False


def _cross_encoder_rerank(query: str, documents, top_k: int = 5):
    """Rerank documents using a lightweight cross-encoder via fastembed."""
    if not documents:
        return documents

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    except Exception:
        # Fallback: return original order if reranker fails
        return documents[:top_k]


class ToolFactory:

    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            # Over-fetch then rerank for better precision
            fetch_k = max(limit * 3, 15)
            results = self.collection.similarity_search(query, k=fetch_k, score_threshold=0.4)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            # Rerank to get the most relevant results
            results = _cross_encoder_rerank(query, results, top_k=limit)

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])            

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]