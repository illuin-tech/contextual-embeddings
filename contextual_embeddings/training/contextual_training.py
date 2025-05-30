import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import SentenceEvaluator

from ..collators.contextual_collator import ContextualDataCollator
from ..models.long_context_model import LongContextEmbeddingModel
from ..training.contextual_trainer import ContextualTrainer


@dataclass
class ContextualTrainingConfig:
    model: LongContextEmbeddingModel
    exp_name: str = "test_exp"
    n_gpus: int = 1
    training_args: SentenceTransformerTrainingArguments = None
    output_dir: str = None
    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None
    evaluator: Optional[SentenceEvaluator] = None
    run_train: bool = True
    multi_ctx_training: bool = True
    wandb_project: str = "long-context-model"
    wandb_group: str = "main_exp"
    loss_type: str = "inbatch_inseq"
    add_prefixes: bool = False
    colbert_tokenize: bool = False

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        self.base_output_dir = self.output_dir
        self.output_dir = os.path.join(self.output_dir, self.exp_name)

        if os.path.exists(self.output_dir):
            dt = datetime.now().strftime("%Y%m%d%H%M%S")
            self.output_dir += f"_{dt}"

        if self.training_args is None:
            self.training_args = SentenceTransformerTrainingArguments(output_dir=self.output_dir)
        elif self.training_args.output_dir is None:
            self.training_args.output_dir = self.output_dir

        if self.training_args.run_name is None:
            self.training_args.run_name = self.exp_name

        # cast if string
        if isinstance(self.training_args.learning_rate, str):
            self.training_args.learning_rate = float(self.training_args.learning_rate)

        self.training_args.remove_unused_columns = False

    def set_exp_name(self, exp_name: str):
        self.exp_name = exp_name
        self.output_dir = os.path.join(self.base_output_dir, self.exp_name)
        self.training_args.output_dir = self.output_dir
        self.training_args.run_name = self.exp_name


class ContextualTraining:
    def __init__(self, config: ContextualTrainingConfig):
        self.config = config
        self.model = config.model

    def train(self):
        """
        Train the model using the provided configuration.
        """
        trainer = ContextualTrainer(
            training_loss_type=self.config.loss_type,
            model=self.model,
            tokenizer=self.model.base_model.tokenizer,
            args=self.config.training_args,
            train_dataset=self.config.train_dataset,
            eval_dataset=self.config.eval_dataset,
            evaluator=self.config.evaluator,
            data_collator=ContextualDataCollator(
                tokenize_fn=self.model.tokenize,
                is_multi_ctx_training=self.config.multi_ctx_training,
                sep_token=self.model.base_model.tokenizer.sep_token,
                add_prefixes=self.config.add_prefixes,
                colbert_tokenize=self.config.colbert_tokenize,
            ),
        )

        trainer.train()

    def save(self, config_file):
        """
        Save the trained model and configuration.
        Args:
          config_file: Path to the configuration file to be copied.
        """
        # save model
        self.model.save(self.config.output_dir)

        # copy-paste the yml file with os
        # ugly but no other way since the file is formatted for the configue library
        # so not fully complying to the yaml syntax (hence not supported by e.g. pyyaml)
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")
