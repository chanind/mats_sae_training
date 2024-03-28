from typing import Iterable

from transformer_lens import HookedTransformer

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sae_group import SAEGroup
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.train_sae_on_language_model import train_sae_group_on_language_model
from sae_training.utils import LMSparseAutoencoderSessionloader


def language_model_sae_group_runner_from_pretrained(path: str) -> SAEGroup:
    (
        model,
        sae_group,
        activations_loader,
    ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(path)
    return execute_language_model_sae_group_runner(model, sae_group, activations_loader)


def language_model_sae_runner(cfg: LanguageModelSAERunnerConfig) -> SparseAutoencoder:
    sae_group = language_model_sae_group_runner(cfg)
    return sae_group.autoencoders[0]


def language_model_sae_group_runner(
    cfg: LanguageModelSAERunnerConfig | Iterable[LanguageModelSAERunnerConfig],
) -> SAEGroup:
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sae_group, activations_loader = loader.load_session()
    return execute_language_model_sae_group_runner(model, sae_group, activations_loader)


def execute_language_model_sae_group_runner(
    model: HookedTransformer, sae_group: SAEGroup, activations_loader: ActivationsStore
):
    shared_config = sae_group.shared_config

    if shared_config.log_to_wandb:
        wandb.init(
            project=shared_config.wandb_project,
            config=sae_group.get_combined_sae_configs(),
            name=shared_config.run_name,
        )

    # train SAE
    train_output = train_sae_group_on_language_model(
        model,
        sae_group,
        activations_loader,
        n_checkpoints=shared_config.n_checkpoints,
        batch_size=shared_config.train_batch_size,
        feature_sampling_window=shared_config.feature_sampling_window,
        use_wandb=shared_config.log_to_wandb,
        wandb_log_frequency=shared_config.wandb_log_frequency,
    )

    if shared_config.log_to_wandb:
        wandb.finish()

    return train_output.sae_group
