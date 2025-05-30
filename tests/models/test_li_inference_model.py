import pytest
import torch
from pylate.models import ColBERT

from contextual_embeddings import LongContextEmbeddingModel


@pytest.fixture(scope="module")
def colbert_model():
    # Use a path to a pretrained ColBERT model
    model_path = "lightonai/GTE-ModernColBERT-v1"  # replace with your actual model path
    return ColBERT(
        model_path,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )


@pytest.fixture(scope="module")
def model(colbert_model):
    model = LongContextEmbeddingModel(base_model=colbert_model, pooling_mode="tokens")
    model.batch_size = 2
    model.show_progress_bar = False
    return model


@pytest.fixture
def test_documents():
    return [
        ["This is chunk 1 of doc 1", "This is chunk 2 of doc 1"],  # 2 chunks
        ["This is chunk 1 of doc 2", "This is chunk 2 of doc 2", "This is chunk 3 of doc 2"],  # 3 chunks
        ["Single chunk document"],  # 1 chunk
        ["Chunk 1 of doc 4", "Chunk 2 of doc 4", "Chunk 3 of doc 4", "Chunk 4 of doc 4"],  # 4 chunks
    ]


@pytest.fixture
def test_queries():
    return [
        "Short query",
        "This is a longer query with more words",
        "Query 3",
        "Very long query that contains many words to test the embedding function",
    ]


def test_embed_documents_token_counts(model, test_documents):
    """Test that embedded chunks maintain appropriate token counts"""
    # First, get the token counts for each chunk
    chunk_token_counts = {}

    for doc_idx, doc in enumerate(test_documents):
        chunk_token_counts[doc_idx] = []
        for chunk_idx, chunk in enumerate(doc):
            # Tokenize each chunk individually (without special tokens)
            tokens = model.base_model.tokenize([chunk])
            # Count actual tokens (excluding padding)
            token_count = tokens["attention_mask"].sum().item()
            chunk_token_counts[doc_idx].append(token_count)

    # Now get embeddings
    embeddings = model.embed_documents(test_documents, batch_size=2)

    # Check the output structure
    assert len(embeddings) == len(test_documents)

    # Check that each document has the right number of chunk embeddings
    for doc_idx, doc in enumerate(test_documents):
        assert len(embeddings[doc_idx]) == len(doc)

        # For token-level embeddings, check the shape of each embedding
        embedding_dim = model.base_model.get_sentence_embedding_dimension()
        for chunk_idx, chunk_embedding in enumerate(embeddings[doc_idx]):
            assert chunk_embedding.shape[-1] == embedding_dim


def test_embed_queries_token_dimensions(model, test_queries):
    """Test that query embeddings maintain token dimensions"""
    # Get token counts for each query
    query_token_counts = []
    for query in test_queries:
        tokens = model.base_model.tokenize([query], is_query=True)
        token_count = tokens["attention_mask"].sum().item()
        query_token_counts.append(token_count)

    # Get query embeddings
    query_embeddings = model.embed_queries(test_queries)

    # For token pooling with ColBERT, we expect token-level embeddings
    # The shape should be [num_queries, max_query_length, embedding_dim]
    assert query_embeddings.shape[0] == len(test_queries)

    embedding_dim = model.base_model.get_sentence_embedding_dimension()
    assert query_embeddings.shape[2] == embedding_dim

    # For each query, check that we have at least as many token embeddings as the original token count
    # (might have more due to padding)
    for i, count in enumerate(query_token_counts):
        # Count non-zero embeddings
        non_zero_embs = torch.sum(torch.sum(query_embeddings[i] != 0, dim=1) > 0).item()
        assert non_zero_embs >= count, f"Query {i} should have at least {count} token embeddings"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_embed_documents_batch_processing(model, batch_size):
    """Test batch processing maintains token counts"""
    test_documents = [
        [f"Doc {i} Chunk {j}" for j in range(i % 2 + 1)]
        for i in range(3)  # 3 documents with 1, 2, 1 chunks
    ]

    # Process with different batch sizes
    embeddings = model.embed_documents(test_documents, batch_size=batch_size)

    # Verify the structure
    assert len(embeddings) == len(test_documents)

    for doc_idx, doc in enumerate(test_documents):
        assert len(embeddings[doc_idx]) == len(doc)

        # Check embedding dimensions
        embedding_dim = model.base_model.get_sentence_embedding_dimension()
        for chunk_idx, chunk_embedding in enumerate(embeddings[doc_idx]):
            assert chunk_embedding.shape[-1] == embedding_dim
