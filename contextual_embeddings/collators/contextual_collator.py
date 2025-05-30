from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sentence_transformers.data_collator import SentenceTransformerDataCollator
from torch import Tensor


@dataclass
class ContextualDataCollator(SentenceTransformerDataCollator):
    is_multi_ctx_training: bool = field(default=True, kw_only=True)
    add_prefixes: bool = field(default=False, kw_only=True)
    sep_token: Optional[str] = field(default=None, kw_only=True)
    colbert_tokenize: Optional[bool] = field(default=False, kw_only=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        # We should always be able to return a loss, label or not:
        batch = {}

        # tokenize the queries

        # merge the lists of queries into one list for the tokenizer
        all_queries = [query for row in features for query in row["queries"]]

        # TODO: put prefixes as parameters
        if self.add_prefixes:
            all_queries = [f"search_query: {query}" for query in all_queries]

        tokenized = self.tokenize_fn(all_queries)
        for key, value in tokenized.items():
            batch[f"queries_{key}"] = value

        # tokenize the documents
        if self.is_multi_ctx_training:
            concatenated_docs = [self.sep_token.join(row["docs_list"]) for row in features]
            if self.add_prefixes:
                concatenated_docs = [f"search_document: {doc}" for doc in concatenated_docs]

            if self.colbert_tokenize:
                tokenized = self.tokenize_fn(concatenated_docs, is_query=False)
            else:
                tokenized = self.tokenize_fn(concatenated_docs)
        else:
            single_docs = [row["docs_list"] for row in features]
            batch["n_docs_per_sample"] = [len(docs) for docs in single_docs]
            flattened_docs = [doc for sublist in single_docs for doc in sublist]

            if self.add_prefixes:
                flattened_docs = [f"search_document: {doc}" for doc in flattened_docs]

            # TODO: add assertion to check that max idx in queries_chunk_indices does not exceed n_docs
            assert len(single_docs) >= 1, "Single context collation requires at least one document"

            if self.colbert_tokenize:
                tokenized = self.tokenize_fn(flattened_docs, is_query=False)
            else:
                tokenized = self.tokenize_fn(flattened_docs)

        for key, value in tokenized.items():
            batch[f"docs_{key}"] = value

        # merge the queries_chunk_ids into one tensor
        batch["queries_chunk_indices"] = [row["queries_chunk_ids"] for row in features]

        batch["add_prefixes"] = self.add_prefixes

        return batch
