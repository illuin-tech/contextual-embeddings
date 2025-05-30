import pytest
import torch
from sentence_transformers import (
    SentenceTransformer,
)

from contextual_embeddings import LongContextEmbeddingModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS = 10
DOC_TOY_STR = "This is an example document."
QUERY_TOY_STR = "toy query"


@pytest.fixture(scope="module")
def model_str():
    return "nomic-ai/modernbert-embed-base"


@pytest.fixture(scope="module")
def base_model(model_str):
    model = SentenceTransformer(model_str)
    yield model
    del model


@pytest.fixture(scope="module")
def hidden_dim(base_model: SentenceTransformer):
    return base_model.get_sentence_embedding_dimension()


@pytest.fixture(scope="module")
def doc_str(base_model):
    return base_model.tokenizer.sep_token.join([DOC_TOY_STR for _ in range(N_DOCS)])


@pytest.fixture(scope="module")
def model(base_model):
    return LongContextEmbeddingModel(base_model).to(DEVICE)


def test_forward_multi_docs(model, doc_str):
    doc_inputs = model.tokenize([doc_str])
    query_inputs = model.tokenize([QUERY_TOY_STR])
    qc_indices = [[1]]
    batch = {
        "docs_inputs": doc_inputs,
        "query_inputs": query_inputs,
    }
    outputs = model(batch, queries_chunk_indices=qc_indices)

    assert "loss" in outputs


def test_late_chunking_pooling(model: LongContextEmbeddingModel, doc_str, hidden_dim):
    inputs = model.tokenize([doc_str])
    input_ids = inputs["input_ids"]
    token_embeddings = torch.randn((input_ids.shape[0], input_ids.shape[1], hidden_dim))
    pooled_chunks, padding_mask = model._late_chunking_pooling(input_ids, token_embeddings)

    assert pooled_chunks.shape == (1, N_DOCS, hidden_dim)
    assert padding_mask.shape == (1, N_DOCS)

    # batch test
    batch_size = 4
    other_str = model.base_model.tokenizer.sep_token.join([DOC_TOY_STR for _ in range(N_DOCS - 1)])

    inputs = model.tokenize([doc_str, other_str, doc_str, other_str])
    input_ids = inputs["input_ids"]
    token_embeddings = torch.randn((input_ids.shape[0], input_ids.shape[1], hidden_dim))
    pooled_chunks, padding_mask = model._late_chunking_pooling(input_ids, token_embeddings)
    assert pooled_chunks.shape == (batch_size, N_DOCS, hidden_dim)
    assert padding_mask.shape == (batch_size, N_DOCS)
    assert padding_mask.sum() == 2
