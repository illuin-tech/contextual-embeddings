import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

N_SPECIAL_TOKENS = 3  # to be safe, we imagine the tokenizer will add 3 special tokens


def get_dataset_from_beir_format(
    queries: Dataset,
    chunks: Dataset,
    sep_token: str,
) -> Dataset:
    # Extract ids for both queries and chunks (format is always docID_chunkID)
    def extract_ids(sample):
        split = sample["chunk_id"].split("_")
        sample["doc_id"] = split[0]
        sample["chunk_id"] = int(split[1])
        return sample

    chunks = chunks.map(extract_ids)
    queries = queries.map(extract_ids)

    df_chunks = pd.DataFrame(chunks)
    grouped_chunks = (
        df_chunks.sort_values(["doc_id", "chunk_id"]).groupby("doc_id").agg({"chunk": list, "chunk_id": list})
    )
    grouped_chunks = grouped_chunks.rename(columns={"chunk": "docs_list"})

    assert grouped_chunks["chunk_id"].apply(lambda x: x == sorted(x)).all()
    grouped_chunks = grouped_chunks.drop(columns=["chunk_id"])

    # concatenate chunks with sep_tokens (keep the list as a separate entry)
    grouped_chunks["docs"] = grouped_chunks["docs_list"].apply(lambda x: sep_token.join(x))

    df_queries = pd.DataFrame(queries)
    df_queries = df_queries[df_queries["query"].apply(lambda x: x.strip() != "")]
    grouped_queries = (
        df_queries.sort_values(["doc_id", "chunk_id"])
        .groupby("doc_id")
        .agg({"query": list, "chunk_id": list, "answer": list})
    )
    assert grouped_queries["chunk_id"].apply(lambda x: x == sorted(x)).all()
    grouped_queries = grouped_queries.rename(columns={"query": "queries", "chunk_id": "queries_chunk_ids"})

    df_dataset = grouped_queries.join(grouped_chunks, how="inner")

    assert len(df_dataset) == len(grouped_queries), "All queries should have a corresponding document."

    dataset = Dataset.from_pandas(df_dataset)

    return dataset


def get_chunked_mldr_st(path, base_model, split="train", all_queries=True, filter_long_samples=True):
    ds_docs = load_dataset(path, "documents", split=split)

    if filter_long_samples:

        def extract_n_tokens(sample):
            sample["n_tokens"] = len(base_model.tokenizer(sample["chunk"]).input_ids)
            return sample

        ds_docs = ds_docs.map(extract_n_tokens)
        ds_docs = ds_docs.filter(lambda x: x["n_tokens"] < base_model.max_seq_length - N_SPECIAL_TOKENS)

    ds_queries = load_dataset(path, "queries", split=split)
    if all_queries:
        ds_synthetic = load_dataset(path, "synthetic_queries", split=split)
        ds_queries = concatenate_datasets([ds_queries, ds_synthetic])

    chunk_ids_mapping = {s["chunk_id"]: i for i, s in enumerate(ds_docs)}

    dataset_queries = []
    dataset_docs = []
    for sample in tqdm(ds_queries):
        if sample["chunk_id"] in chunk_ids_mapping:
            dataset_queries.append(sample["query"])
            dataset_docs.append(ds_docs[chunk_ids_mapping[sample["chunk_id"]]]["chunk"])

    dataset = Dataset.from_dict({"queries": dataset_queries, "docs": dataset_docs})
    return dataset


def create_contextual_dataset(path, base_model, split="train", all_queries=True, filter_long_samples=True):
    ds_docs = load_dataset(path, "documents", split=split)
    ds_queries = load_dataset(path, "queries", split=split)
    if all_queries:
        ds_synthetic = load_dataset(path, "synthetic_queries", split=split)
        ds_queries = concatenate_datasets([ds_queries, ds_synthetic])

    print(f"Number of total queries: {len(ds_queries)}")
    dataset = get_dataset_from_beir_format(ds_queries, ds_docs, base_model.tokenizer.sep_token)

    if filter_long_samples:
        dataset = dataset.filter(
            lambda x: len(base_model.tokenizer(x["docs"]).input_ids) < base_model.max_seq_length - N_SPECIAL_TOKENS
        )

    def remove_bad_queries(sample):
        mask = [qc_id < len(sample["docs_list"]) for qc_id in sample["queries_chunk_ids"]]
        sample["queries_chunk_ids"] = [qc_id for qc_id, m in zip(sample["queries_chunk_ids"], mask) if m]
        sample["queries"] = [q for q, m in zip(sample["queries"], mask) if m]
        return sample

    print(f"Number of queries after filtering: {sum([len(q) for q in dataset['queries']])}")
    dataset = dataset.map(remove_bad_queries)
    print(f"Number of queries after removing bad queries: {sum([len(q) for q in dataset['queries']])}")

    return dataset


def get_nomic_clusters_dataset(path, base_model, split="train", num_proc=64):
    dataset = load_dataset(path, split=split)

    def extract_n_tokens(sample):
        sample["n_tokens"] = len(base_model.tokenizer(sample["docs"]).input_ids)
        sample["queries_chunk_ids"] = list(range(len(sample["queries"])))
        return sample

    dataset = dataset.map(extract_n_tokens)

    dataset = dataset.filter(
        lambda x: x["n_tokens"] < base_model.max_seq_length - N_SPECIAL_TOKENS,
    )

    return dataset


def get_long_context_dataset(base_model, base_path="./data_dir/", split="train", return_all=False):
    ds_mldr_big = create_contextual_dataset(f"{base_path}/chunked-mldr-big", base_model, split=split)
    ds_narrative_qa = create_contextual_dataset(f"{base_path}/narrative_qa", base_model, split=split, all_queries=False)
    ds_squad = create_contextual_dataset(f"{base_path}/squad", base_model, split=split, all_queries=False)
    full_dataset = concatenate_datasets([ds_mldr_big, ds_narrative_qa, ds_squad])

    if return_all:
        return ds_mldr_big, ds_narrative_qa, ds_squad, full_dataset

    return full_dataset


def get_smaller_chunks_dataset(base_model, base_path="./data_dir/", split="train"):
    ds_mldr_big = create_contextual_dataset(f"{base_path}/chunked-mldr-big-100", base_model, split=split)
    ds_squad = create_contextual_dataset(
        f"{base_path}/squad-chunked-par-100", base_model, split=split, all_queries=False
    )
    full_dataset = concatenate_datasets([ds_mldr_big, ds_squad])
    return full_dataset


def get_mixed_granularity_dataset(base_model, base_path="./data_dir/", split="train"):
    ds_mldr_big = create_contextual_dataset(f"{base_path}/chunked-mldr-big", base_model, split=split)
    ds_narrative_qa = create_contextual_dataset(f"{base_path}/narrative_qa", base_model, split=split, all_queries=False)
    ds_squad = create_contextual_dataset(f"{base_path}/squad", base_model, split=split, all_queries=False)
    ds_mldr_big_chunked = create_contextual_dataset(f"{base_path}/chunked-mldr-big-100", base_model, split=split)
    ds_squad_chunked = create_contextual_dataset(
        f"{base_path}/squad-chunked-par-100", base_model, split=split, all_queries=False
    )
    full_dataset = concatenate_datasets([ds_mldr_big, ds_narrative_qa, ds_squad, ds_mldr_big_chunked, ds_squad_chunked])
    return full_dataset


def get_nomic_st(path, base_model, split="train"):
    # nomic_embed_supervised_clustered
    dataset = load_dataset(path, split=split)

    def extract_n_tokens(sample):
        sample["n_tokens"] = len(base_model.tokenizer(sample["document"]).input_ids)
        return sample

    dataset = dataset.map(extract_n_tokens)

    dataset = dataset.filter(
        lambda x: x["n_tokens"] < base_model.max_seq_length - N_SPECIAL_TOKENS,
    )

    # only keep query and document fields
    dataset = Dataset.from_dict({"queries": dataset["query"], "docs": dataset["document"]})

    return dataset
