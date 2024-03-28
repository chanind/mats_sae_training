import tempfile

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import LMSparseAutoencoderSessionloader

TEST_MODEL = "tiny-stories-1M"
TEST_DATASET = "roneneldan/TinyStories"


@pytest.fixture
def cfg():
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    cfg = LanguageModelSAERunnerConfig(
        model_name=TEST_MODEL,
        hook_point_template="blocks.{layer}.hook_mlp_out",
        hook_point_layer=0,
        dataset_path=TEST_DATASET,
        is_dataset_tokenized=False,
        d_in=64,
        expansion_factor=2,
        l1_coefficient=2e-3,
        lr=2e-4,
        train_batch_size=512,
        context_size=64,
        feature_sampling_window=50,
        dead_feature_threshold=1e-7,
        n_batches_in_buffer=2,
        total_training_tokens=1_000_000,
        store_batch_size=128,
        log_to_wandb=False,
        wandb_project="test_project",
        wandb_entity="test_entity",
        wandb_log_frequency=10,
        device="cpu",
        checkpoint_path="test/checkpoints",
        dtype=torch.float32,
        use_cached_activations=False,
        hook_point_head_index=None,
    )

    return cfg


def test_LMSparseAutoencoderSessionloader_init(cfg: LanguageModelSAERunnerConfig):
    loader = LMSparseAutoencoderSessionloader(cfg)
    assert loader.cfgs == cfg


def test_LMSparseAutoencoderSessionloader_load_session(
    cfg: LanguageModelSAERunnerConfig,
):
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sae_group, activations_loader = loader.load_session()

    assert isinstance(model, HookedTransformer)
    assert isinstance(sae_group.autoencoders[0], SparseAutoencoder)
    assert isinstance(activations_loader, ActivationsStore)


def test_LMSparseAutoencoderSessionloader_load_session_from_trained(
    cfg: LanguageModelSAERunnerConfig,
):
    loader = LMSparseAutoencoderSessionloader(cfg)
    _, sae_group, _ = loader.load_session()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tempfile_path = f"{tmpdirname}/test.pt"
        sae_group.save_model(tempfile_path)

        (
            _,
            new_sae_group,
            _,
        ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(tempfile_path)
    for sae in new_sae_group.autoencoders:
        sae.cfg.device = torch.device("cpu")
    new_sae_group.to("cpu")
    for new_sae, old_sae in zip(new_sae_group.autoencoders, sae_group.autoencoders):
        assert new_sae.cfg == old_sae.cfg
    # assert weights are the same
    new_parameters = dict(new_sae_group.autoencoders[0].named_parameters())
    for name, param in sae_group.autoencoders[0].named_parameters():
        assert torch.allclose(param, new_parameters[name])
