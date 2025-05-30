from typing import Any

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from torch import Tensor


class ContextualTrainer(SentenceTransformerTrainer):
    def __init__(self, training_loss_type: str = "inbatch_inseq", **super_kwargs):
        super().__init__(**super_kwargs)
        self.training_loss_type = training_loss_type

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        features, _ = self.collect_features(inputs)

        # Pass everything to the model and compute the loss in the model directly
        features_dict = {
            "query_inputs": {k: v for k, v in features[0].items()},
            "docs_inputs": {k: v for k, v in features[1].items()},
        }

        model_outputs = model(
            features_dict,
            queries_chunk_indices=inputs["queries_chunk_indices"],
            n_docs_per_sample=inputs.get("n_docs_per_sample"),
            loss_type=self.training_loss_type,
            add_prefixes=inputs.get("add_prefixes", False),
        )
        loss = model_outputs["loss"]

        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}
        return loss
