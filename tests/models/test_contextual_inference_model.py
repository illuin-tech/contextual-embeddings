import pytest
import torch
from sentence_transformers import SentenceTransformer

from contextual_embeddings import LongContextEmbeddingModel


@pytest.fixture(scope="module")
def model():
    # Use a small model for testing
    base_model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    model = LongContextEmbeddingModel(base_model=base_model, normalize_embeddings=True, pooling_mode="average")
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


@pytest.fixture(scope="module")
def model_with_prefix():
    base_model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    model = LongContextEmbeddingModel(
        base_model=base_model, normalize_embeddings=True, pooling_mode="average", add_prefix=True
    )
    model.batch_size = 2
    model.show_progress_bar = False
    return model


def test_embed_documents(model, test_documents):
    # Get embeddings
    embeddings = model.embed_documents(test_documents, batch_size=2)

    # Test the output structure
    assert len(embeddings) == len(test_documents), (
        "Number of document embeddings should match number of input documents"
    )

    # Check that each document has the right number of chunk embeddings
    for i, doc in enumerate(test_documents):
        assert len(embeddings[i]) == len(doc), (
            f"Document {i} should have {len(doc)} chunk embeddings but has {len(embeddings[i])}"
        )

        # Check embedding dimensions
        embedding_dim = model.base_model.get_sentence_embedding_dimension()
        for chunk_embedding in embeddings[i]:
            assert chunk_embedding.shape == (embedding_dim,), f"Chunk embedding dimension should be {embedding_dim}"

            # Check if embeddings are normalized
            if model.normalize_embeddings:
                norm = torch.norm(chunk_embedding).item()
                assert abs(norm - 1.0) < 1e-5, "Embeddings should be normalized to unit length"


def test_embed_queries(model, test_queries):
    # Get query embeddings
    query_embeddings = model.embed_queries(test_queries)

    # Test output structure
    if model.pooling_mode == "average":
        # For average pooling, we expect a 2D tensor
        assert len(query_embeddings) == len(test_queries), (
            "Number of query embeddings should match number of input queries"
        )

        embedding_dim = model.base_model.get_sentence_embedding_dimension()
        assert query_embeddings.shape == (len(test_queries), embedding_dim), (
            f"Query embeddings should have shape ({len(test_queries)}, {embedding_dim})"
        )

        # Check if embeddings are normalized
        if model.normalize_embeddings:
            norms = torch.norm(query_embeddings, dim=1)
            for norm in norms:
                assert abs(norm.item() - 1.0) < 1e-5, "Query embeddings should be normalized to unit length"
    else:
        # For token pooling, we expect padded token embeddings
        assert query_embeddings.shape[0] == len(test_queries), (
            "Number of query embeddings should match number of input queries"
        )

        embedding_dim = model.base_model.get_sentence_embedding_dimension()
        assert query_embeddings.shape[2] == embedding_dim, (
            f"Query token embeddings should have dimension {embedding_dim}"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_embed_documents_with_batch_size(model, batch_size):
    # Create a larger test set to test batch processing
    test_documents = [
        [f"Doc {i} Chunk {j}" for j in range(i % 3 + 1)]
        for i in range(5)  # 5 documents with 1, 2, 3, 1, 2 chunks
    ]

    embeddings = model.embed_documents(test_documents, batch_size=batch_size)

    # Verify all documents have embeddings
    assert len(embeddings) == len(test_documents)

    # Verify each document has the right number of chunk embeddings
    for i, doc in enumerate(test_documents):
        assert len(embeddings[i]) == len(doc), (
            f"Document {i} should have {len(doc)} chunk embeddings with batch size {batch_size}"
        )


def test_embed_documents_with_prefix(model_with_prefix, test_documents):
    # Get embeddings with prefix
    embeddings = model_with_prefix.embed_documents(test_documents, batch_size=2)

    # Test the output structure
    assert len(embeddings) == len(test_documents), (
        "Number of document embeddings should match number of input documents"
    )

    # Check that each document has the right number of chunk embeddings
    for i, doc in enumerate(test_documents):
        assert len(embeddings[i]) == len(doc), (
            f"Document {i} should have {len(doc)} chunk embeddings but has {len(embeddings[i])}"
        )

        # Check embedding dimensions
        embedding_dim = model_with_prefix.base_model.get_sentence_embedding_dimension()
        for chunk_embedding in embeddings[i]:
            assert chunk_embedding.shape == (embedding_dim,), f"Chunk embedding dimension should be {embedding_dim}"

            # Check if embeddings are normalized
            if model_with_prefix.normalize_embeddings:
                norm = torch.norm(chunk_embedding).item()
                assert abs(norm - 1.0) < 1e-5, "Embeddings should be normalized to unit length"


def test_embeddings_are_different_with_prefix(model, model_with_prefix, test_documents, test_queries):
    """Test that embeddings with prefix are different from embeddings without prefix."""
    # Get embeddings without prefix
    doc_embeddings_no_prefix = model.embed_documents(test_documents, batch_size=1)
    query_embeddings_no_prefix = model.embed_queries(test_queries)

    # Get embeddings with prefix
    doc_embeddings_with_prefix = model_with_prefix.embed_documents(test_documents, batch_size=1)
    query_embeddings_with_prefix = model_with_prefix.embed_queries(test_queries)

    # Check documents
    for i in range(len(test_documents)):
        for j in range(len(test_documents[i])):
            # Embeddings should be different when using prefix
            assert not torch.allclose(doc_embeddings_no_prefix[i][j], doc_embeddings_with_prefix[i][j], atol=1e-4), (
                f"Document {i} chunk {j} embeddings should be different with prefix"
            )

    # Check queries (for average pooling)
    if model.pooling_mode == "average":
        for i in range(len(test_queries)):
            assert not torch.allclose(query_embeddings_no_prefix[i], query_embeddings_with_prefix[i], atol=1e-4), (
                f"Query {i} embeddings should be different with prefix"
            )
