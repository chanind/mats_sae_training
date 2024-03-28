from collections.abc import Iterable
from math import ceil

import pytest
import torch
from datasets import Dataset, IterableDataset
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore, ActivationsStoreConfig
from sae_training.config import LanguageModelSAERunnerConfig
from tests.unit.helpers import build_sae_cfg


def tokenize_with_bos(model: HookedTransformer, text: str) -> list[int]:
    assert model.tokenizer is not None
    assert model.tokenizer.bos_token_id is not None
    return [model.tokenizer.bos_token_id] + model.tokenizer.encode(text)


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point_template": "blocks.{layer}.hook_resid_pre",
            "hook_point_layers": [1],
            "d_in": 64,
            "prepend_bos": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point_template": "blocks.{layer}.attn.hook_z",
            "hook_point_layers": [1],
            "d_in": 64,
            "prepend_bos": True,
        },
        {
            "model_name": "gelu-2l",
            "dataset_path": "NeelNanda/c4-tokenized-2b",
            "tokenized": True,
            "hook_point_template": "blocks.{layer}.hook_resid_pre",
            "hook_point_layers": [1],
            "d_in": 512,
            "prepend_bos": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
            "tokenized": True,
            "hook_point_template": "blocks.{layer}.hook_resid_pre",
            "hook_point_layers": [1],
            "d_in": 768,
            "prepend_bos": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "Skylion007/openwebtext",
            "tokenized": False,
            "hook_point_template": "blocks.{layer}.hook_resid_pre",
            "hook_point_layers": [1],
            "d_in": 768,
            "prepend_bos": True,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-attn-out",
        "gelu-2l-tokenized",
        "gpt2-tokenized",
        "gpt2",
    ],
)
def cfg(request: pytest.FixtureRequest) -> LanguageModelSAERunnerConfig:
    # This function will be called with each parameter set
    params = request.param
    return build_sae_cfg(**params)


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig):
    return HookedTransformer.from_pretrained(cfg.model_name, device="cpu")


@pytest.fixture
def activation_store(cfg: LanguageModelSAERunnerConfig, model: HookedTransformer):
    store_cfg = ActivationsStoreConfig.from_sae_runner_config(cfg)
    return ActivationsStore(store_cfg, model)


def test_activations_store__init__(
    cfg: LanguageModelSAERunnerConfig, model: HookedTransformer
):
    store_cfg = ActivationsStoreConfig.from_sae_runner_config(cfg)
    store = ActivationsStore(store_cfg, model)

    assert store.model == model

    assert isinstance(store.dataset, IterableDataset)
    assert isinstance(store.iterable_dataset, Iterable)

    # I expect the dataloader to be initialised
    assert hasattr(store, "dataloader")

    # I expect the buffer to be initialised
    assert hasattr(store, "storage_buffer")

    # the rest is in the dataloader.
    expected_size = (
        cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer // 2
    )
    assert store.storage_buffer.shape == (expected_size, 1, cfg.d_in)


def test_activations_store__get_batch_tokens(activation_store: ActivationsStore):
    batch = activation_store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (
        activation_store.cfg.store_batch_size,
        activation_store.cfg.context_size,
    )
    assert batch.device == activation_store.cfg.device


def test_activations_score_get_next_batch(
    model: HookedTransformer, activation_store: ActivationsStore
):

    batch = activation_store.get_batch_tokens()
    assert batch.shape == (
        activation_store.cfg.store_batch_size,
        activation_store.cfg.context_size,
    )

    # if model.tokenizer.bos_token_id is not None:
    #     torch.testing.assert_close(
    #         batch[:, 0], torch.ones_like(batch[:, 0]) * model.tokenizer.bos_token_id
    #     )


def test_activations_store__get_activations(activation_store: ActivationsStore):
    batch = activation_store.get_batch_tokens()
    activations = activation_store.get_activations(batch)

    cfg = activation_store.cfg
    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (cfg.store_batch_size, cfg.context_size, 1, cfg.d_in)
    assert activations.device == cfg.device


def test_activations_store__get_activations_head_hook(ts_model: HookedTransformer):
    cfg = build_sae_cfg(
        hook_point_template="blocks.{layer}.attn.hook_q",
        hook_point_head_index=2,
        hook_point_layer=1,
        d_in=4,
    )
    store_cfg = ActivationsStoreConfig.from_sae_runner_config(cfg)
    activation_store_head_hook = ActivationsStore(store_cfg, ts_model)
    batch = activation_store_head_hook.get_batch_tokens()
    activations = activation_store_head_hook.get_activations(batch)

    cfg = activation_store_head_hook.cfg
    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (cfg.store_batch_size, cfg.context_size, 1, cfg.d_in)
    assert activations.device == cfg.device


def test_activations_store__get_buffer(activation_store: ActivationsStore):
    n_batches_in_buffer = 3
    buffer = activation_store.get_buffer(n_batches_in_buffer)

    cfg = activation_store.cfg
    assert isinstance(buffer, torch.Tensor)
    buffer_size_expected = cfg.store_batch_size * cfg.context_size * n_batches_in_buffer

    assert buffer.shape == (buffer_size_expected, 1, cfg.d_in)
    assert buffer.device == cfg.device


# 12 is divisible by the length of "hello world", 11 and 13 are not
@pytest.mark.parametrize("context_size", [11, 12, 13])
def test_activations_store__get_batch_tokens__fills_the_context_separated_by_bos(
    ts_model: HookedTransformer, context_size: int
):
    assert ts_model.tokenizer is not None
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )
    cfg = build_sae_cfg(
        store_batch_size=2,
        context_size=context_size,
    )
    store_cfg = ActivationsStoreConfig.from_sae_runner_config(cfg)

    activation_store = ActivationsStore(
        store_cfg, ts_model, dataset=dataset, create_dataloader=False
    )
    encoded_text = tokenize_with_bos(ts_model, "hello world")
    tokens = activation_store.get_batch_tokens()
    assert tokens.shape == (2, context_size)  # batch_size x context_size
    all_expected_tokens = (encoded_text * ceil(2 * context_size / len(encoded_text)))[
        : 2 * context_size
    ]
    expected_tokens1 = all_expected_tokens[:context_size]
    expected_tokens2 = all_expected_tokens[context_size:]
    if expected_tokens2[0] != ts_model.tokenizer.bos_token_id:
        expected_tokens2 = [ts_model.tokenizer.bos_token_id] + expected_tokens2[:-1]
    assert tokens[0].tolist() == expected_tokens1
    assert tokens[1].tolist() == expected_tokens2


def test_activations_store__get_next_dataset_tokens__tokenizes_each_example_in_order(
    ts_model: HookedTransformer,
):
    cfg = build_sae_cfg()
    store_cfg = ActivationsStoreConfig.from_sae_runner_config(cfg)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore(
        store_cfg, ts_model, dataset=dataset, create_dataloader=False
    )

    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world1"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world2"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world3"
    )
