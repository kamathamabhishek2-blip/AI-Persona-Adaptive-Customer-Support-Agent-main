# src/rag_pipeline.py

import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import time

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

RAG_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "all-MiniLM-L6-v2",
    "collection_name": "support_kb",
    "vector_db_path": "./chroma_db",
    "data_folder": "data",
    "top_k": 3,
    "batch_size": 32,
    "max_file_size_mb": 50,
    "supported_extensions": [".txt", ".md", ".pdf"],
    "cache_embeddings": True,
    "enable_progress": True,
}

# ============================================
# CLIENT INITIALIZATION (Lazy)
# ============================================

_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {RAG_CONFIG['embedding_model']}")
        _embedding_model = SentenceTransformer(RAG_CONFIG['embedding_model'])
    return _embedding_model


def get_chroma_client():
    """Lazy load ChromaDB client."""
    global _chroma_client, _collection
    if _chroma_client is None:
        logger.info(f"Connecting to ChromaDB: {RAG_CONFIG['vector_db_path']}")
        _chroma_client = chromadb.PersistentClient(path=RAG_CONFIG['vector_db_path'])
        _collection = _chroma_client.get_or_create_collection(
            name=RAG_CONFIG['collection_name']
        )
    return _chroma_client, _collection


def get_collection():
    """Get the ChromaDB collection."""
    _, collection = get_chroma_client()
    return collection


# ============================================
# EMBEDDING CACHE
# ============================================

_embedding_cache = {}

def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding if available."""
    if not RAG_CONFIG["cache_embeddings"]:
        return None
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return _embedding_cache.get(text_hash)


def set_cached_embedding(text: str, embedding: List[float]) -> None:
    """Cache embedding for future use."""
    if not RAG_CONFIG["cache_embeddings"]:
        return
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if len(_embedding_cache) < 10000:  # Limit cache size
        _embedding_cache[text_hash] = embedding


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
    logger.info("Embedding cache cleared")


# ============================================
# DOCUMENT LOADING
# ============================================

def load_documents(data_folder: str = None, max_size_mb: int = None) -> List[Dict]:
    """
    Load all documents from the data folder with validation.
    """
    if data_folder is None:
        data_folder = RAG_CONFIG["data_folder"]
    
    if max_size_mb is None:
        max_size_mb = RAG_CONFIG["max_file_size_mb"]
    
    documents = []
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        logger.warning(f"Data folder created: {data_folder}")
        return documents
    
    files = [f for f in os.listdir(data_folder) 
             if any(f.endswith(ext) for ext in RAG_CONFIG["supported_extensions"])]
    
    if not files:
        logger.warning(f"No supported documents found in {data_folder}")
        return documents
    
    logger.info(f"Found {len(files)} documents to process")
    
    for idx, file in enumerate(files, 1):
        filepath = os.path.join(data_folder, file)
        
        # Check file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.warning(f"Skipping {file}: {file_size_mb:.1f}MB > {max_size_mb}MB limit")
            continue
        
        try:
            if file.endswith(".txt") or file.endswith(".md"):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if text.strip():
                        documents.append({
                            "source": file,
                            "content": text,
                            "type": "text",
                            "size_mb": file_size_mb,
                            "loaded_at": datetime.now().isoformat()
                        })
                        logger.debug(f"Loaded: {file} ({len(text)} chars)")
            
            elif file.endswith(".pdf"):
                try:
                    reader = PdfReader(filepath)
                    full_text = ""
                    page_count = len(reader.pages)
                    
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            full_text += f"\n--- Page {page_num} of {page_count} ---\n{text}"
                    
                    if full_text.strip():
                        documents.append({
                            "source": file,
                            "content": full_text,
                            "type": "pdf",
                            "pages": page_count,
                            "size_mb": file_size_mb,
                            "loaded_at": datetime.now().isoformat()
                        })
                        logger.debug(f"Loaded: {file} ({page_count} pages, {len(full_text)} chars)")
                    
                except Exception as e:
                    logger.error(f"Error reading PDF {file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


# ============================================
# CHUNKING
# ============================================

def split_documents(documents: List[Dict], chunk_size: int = None, 
                   chunk_overlap: int = None) -> List[Dict]:
    """
    Split documents into semantic chunks with overlap.
    """
    if chunk_size is None:
        chunk_size = RAG_CONFIG["chunk_size"]
    
    if chunk_overlap is None:
        chunk_overlap = RAG_CONFIG["chunk_overlap"]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = []
    chunk_id = 0
    
    for doc_idx, doc in enumerate(documents):
        text = doc["content"]
        
        # Skip empty documents
        if not text or not text.strip():
            continue
        
        split_texts = splitter.split_text(text)
        
        for chunk_text in split_texts:
            if chunk_text.strip():
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "source": doc["source"],
                    "doc_type": doc.get("type", "unknown"),
                    "page": doc.get("page", None),
                    "chunk_index": chunk_id,
                    "doc_index": doc_idx,
                    "created_at": datetime.now().isoformat()
                })
                chunk_id += 1
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


# ============================================
# EMBEDDING STORAGE
# ============================================

def store_embeddings(chunks: List[Dict], batch_size: int = None) -> int:
    """
    Store document chunks and their embeddings in ChromaDB with batching.
    """
    if not chunks:
        return 0
    
    if batch_size is None:
        batch_size = RAG_CONFIG["batch_size"]
    
    model = get_embedding_model()
    collection = get_collection()
    
    total_chunks = len(chunks)
    stored_count = 0
    
    logger.info(f"Generating embeddings for {total_chunks} chunks...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [chunk["text"] for chunk in batch]
        batch_ids = [chunk["id"] for chunk in batch]
        batch_metadatas = []
        
        # Generate embeddings with cache check
        embeddings = []
        for text in batch_texts:
            cached = get_cached_embedding(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                emb = model.encode(text).tolist()
                set_cached_embedding(text, emb)
                embeddings.append(emb)
        
        # Build metadata
        for chunk in batch:
            metadata = {
                "source": chunk["source"],
                "doc_type": chunk.get("doc_type", "unknown"),
                "chunk_id": chunk["id"],
                "chunk_index": chunk.get("chunk_index", 0),
                "doc_index": chunk.get("doc_index", 0),
            }
            if chunk.get("page"):
                metadata["page"] = str(chunk["page"])
            if chunk.get("created_at"):
                metadata["created_at"] = chunk["created_at"]
            
            batch_metadatas.append(metadata)
        
        # Add to ChromaDB
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas
            )
            stored_count += len(batch)
            
            if RAG_CONFIG["enable_progress"]:
                logger.info(f"Progress: {stored_count}/{total_chunks} chunks stored")
                
        except Exception as e:
            logger.error(f"Error storing batch: {e}")
            continue
    
    logger.info(f"Successfully stored {stored_count}/{total_chunks} chunks")
    return stored_count


# ============================================
# RETRIEVAL
# ============================================

def preprocess_query(query: str) -> str:
    """
    Clean and normalize query for retrieval.
    """
    if not query:
        return ""
    
    # Remove excessive whitespace
    cleaned = ' '.join(query.strip().split())
    
    # Truncate if too long
    if len(cleaned) > 500:
        cleaned = cleaned[:500]
    
    return cleaned


def retrieve_chunks(query: str, top_k: int = None) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query.
    """
    if top_k is None:
        top_k = RAG_CONFIG["top_k"]
    
    # Clean query
    query = preprocess_query(query)
    if not query:
        return []
    
    # Get model and collection
    model = get_embedding_model()
    collection = get_collection()
    
    try:
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get extra for filtering
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                
                # Filter low quality results
                if similarity >= 0.1:
                    retrieved.append({
                        "text": results["documents"][0][i],
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                        "page": results["metadatas"][0][i].get("page", None),
                        "doc_type": results["metadatas"][0][i].get("doc_type", "unknown"),
                        "distance": distance,
                        "score": similarity,
                        "chunk_id": results["metadatas"][0][i].get("chunk_id", None)
                    })
            
            # Sort by score and return top_k
            retrieved.sort(key=lambda x: x["score"], reverse=True)
            retrieved = retrieved[:top_k]
        
        logger.debug(f"Retrieved {len(retrieved)} chunks for query: {query[:50]}...")
        return retrieved
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


def retrieve_chunks_with_context(query: str, top_k: int = None) -> Tuple[List[Dict], Dict]:
    """
    Retrieve chunks with additional context information.
    """
    chunks = retrieve_chunks(query, top_k=top_k)
    
    context_info = {
        "query": query,
        "total_retrieved": len(chunks),
        "avg_score": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0,
        "sources": list(set(c["source"] for c in chunks)) if chunks else [],
        "confidence": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0
    }
    
    return chunks, context_info


# ============================================
# COLLECTION MANAGEMENT
# ============================================

def get_collection_stats() -> Dict:
    """
    Get statistics about the current collection.
    """
    collection = get_collection()
    count = collection.count()
    
    # Get metadata sample for analysis
    sample_docs = collection.get(limit=5, include=["metadatas"])
    metadata_keys = set()
    if sample_docs and sample_docs["metadatas"]:
        for meta in sample_docs["metadatas"]:
            metadata_keys.update(meta.keys())
    
    return {
        "total_chunks": count,
        "collection_name": RAG_CONFIG["collection_name"],
        "embedding_model": RAG_CONFIG["embedding_model"],
        "chunk_size": RAG_CONFIG["chunk_size"],
        "chunk_overlap": RAG_CONFIG["chunk_overlap"],
        "metadata_fields": list(metadata_keys),
        "is_ready": count > 0,
        "status": "ready" if count > 0 else "empty"
    }


def build_knowledge_base(data_folder: str = None, force_rebuild: bool = False) -> Dict:
    """
    Build the knowledge base by loading, chunking, and embedding documents.
    """
    start_time = time.time()
    
    if data_folder is None:
        data_folder = RAG_CONFIG["data_folder"]
    
    collection = get_collection()
    
    # Check if rebuild is needed
    if not force_rebuild and collection.count() > 0:
        logger.info("Knowledge base already exists")
        return {
            "status": "success",
            "chunks_created": collection.count(),
            "message": "Knowledge base already exists",
            "elapsed_seconds": time.time() - start_time
        }
    
    # Clear existing if rebuilding
    if force_rebuild:
        try:
            chroma_client, _ = get_chroma_client()
            chroma_client.delete_collection(RAG_CONFIG["collection_name"])
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.debug(f"No existing collection to delete: {e}")
        
        # Recreate collection
        global _collection
        _collection = chroma_client.get_or_create_collection(
            name=RAG_CONFIG["collection_name"]
        )
    
    # Load documents
    logger.info(f"Loading documents from {data_folder}...")
    docs = load_documents(data_folder)
    
    if not docs:
        return {
            "status": "error",
            "message": f"No documents found in {data_folder}",
            "elapsed_seconds": time.time() - start_time
        }
    
    # Split into chunks
    logger.info("Chunking documents...")
    chunks = split_documents(docs)
    
    if not chunks:
        return {
            "status": "error",
            "message": "No chunks created from documents",
            "elapsed_seconds": time.time() - start_time
        }
    
    # Store embeddings
    logger.info(f"Storing {len(chunks)} chunks...")
    stored_count = store_embeddings(chunks)
    
    return {
        "status": "success",
        "documents_loaded": len(docs),
        "chunks_created": stored_count,
        "elapsed_seconds": time.time() - start_time,
        "message": f"Successfully indexed {stored_count} chunks from {len(docs)} documents"
    }


# ============================================
# MAINTENANCE FUNCTIONS
# ============================================

def clear_knowledge_base() -> bool:
    """
    Delete all embeddings from the knowledge base.
    """
    try:
        chroma_client, _ = get_chroma_client()
        chroma_client.delete_collection(RAG_CONFIG["collection_name"])
        global _collection
        _collection = chroma_client.get_or_create_collection(
            name=RAG_CONFIG["collection_name"]
        )
        logger.info("Knowledge base cleared")
        return True
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        return False


def get_document_count() -> int:
    """Get total number of chunks in the knowledge base."""
    return get_collection().count()


def is_knowledge_base_ready() -> bool:
    """Check if the knowledge base is ready for queries."""
    return get_collection().count() > 0


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("RAG PIPELINE TEST (OPTIMIZED)")
    print("=" * 60)
    
    # Build knowledge base
    print("\n📚 Building knowledge base...")
    result = build_knowledge_base(force_rebuild=True)
    print(json.dumps(result, indent=2))
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "What is the refund policy?",
        "API authentication error",
        "Service outage impact",
        "Billing issues and payment failures"
    ]
    
    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        chunks, context = retrieve_chunks_with_context(query, top_k=3)
        print(f"   Retrieved: {context['total_retrieved']} chunks")
        print(f"   Confidence: {context['confidence']:.3f}")
        print(f"   Sources: {', '.join(context['sources'])}")
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"\n   {idx}. Source: {chunk['source']}")
            print(f"      Score: {chunk['score']:.3f}")
            print(f"      Preview: {chunk['text'][:100]}...")
        print("-" * 40)
    
    # Performance test
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    import time
    test_query = "How do I reset my password?"
    
    # First query (cold)
    start = time.time()
    chunks, context = retrieve_chunks_with_context(test_query)
    cold_time = time.time() - start
    
    # Second query (warm)
    start = time.time()
    chunks, context = retrieve_chunks_with_context(test_query)
    warm_time = time.time() - start
    
    print(f"Cold query: {cold_time:.3f}s")
    print(f"Warm query: {warm_time:.3f}s")
    print(f"Speedup: {cold_time/warm_time:.1f}x")
    
    # Stats
    stats = get_collection_stats()
    print("\n" + "=" * 60)
    print("COLLECTION STATS")
    print("=" * 60)
    print(json.dumps(stats, indent=2))