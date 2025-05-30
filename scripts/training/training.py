from pathlib import Path

import configue
import typer

from contextual_embeddings.training.contextual_training import (
    ContextualTraining,
    ContextualTrainingConfig,
)


def main(config_file: Path, lambda_seq: float = -1.0) -> None:
    print("Loading config")
    config = configue.load(config_file, sub_path="config")
    print("Creating Setup")
    if isinstance(config, ContextualTrainingConfig):
        app = ContextualTraining(config)
        if lambda_seq > 0:
            app.model.lambda_seq = lambda_seq
            config.set_exp_name(f"{config.exp_name}_{'-'.join(str(lambda_seq).split('.'))}")
    else:
        raise ValueError("Config must be of type ContextualTrainingConfig")

    if config.run_train:
        print("Training model")
        app.train()
        app.save(config_file=config_file)

    """if config.run_eval:
        print("Running evaluation")
        app.eval()
    print("Done!") """


if __name__ == "__main__":
    typer.run(main)
