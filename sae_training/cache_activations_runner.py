import math
import os

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore, ActivationsStoreConfig
from sae_training.config import CacheActivationsRunnerConfig
from sae_training.utils import shuffle_activations_pairwise


def cache_activations_runner(cfg: CacheActivationsRunnerConfig):
    model = HookedTransformer.from_pretrained(cfg.model_name)
    model.to(cfg.device)
    store_cfg = ActivationsStoreConfig(
        hook_point_template=cfg.hook_point_template,
        hook_point_head_index=cfg.hook_point_head_index,
        dataset_path=cfg.dataset_path,
        is_dataset_tokenized=cfg.is_dataset_tokenized,
        context_size=cfg.context_size,
        use_cached_activations=cfg.use_cached_activations,
        cached_activations_path=cfg.cached_activations_path,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        total_training_tokens=cfg.total_training_tokens,
        store_batch_size=cfg.store_batch_size,
        d_in=cfg.d_in,
        device=cfg.device,
        dtype=cfg.dtype,
        train_batch_size=cfg.train_batch_size,
        hook_point_layers=cfg.hook_point_layers,
        prepend_bos=cfg.prepend_bos,
    )
    activations_store = ActivationsStore(store_cfg, model, create_dataloader=False)

    # if the activations directory exists and has files in it, raise an exception
    store_cfg = activations_store.cfg
    assert store_cfg.cached_activations_path is not None
    if os.path.exists(store_cfg.cached_activations_path):
        if len(os.listdir(store_cfg.cached_activations_path)) > 0:
            raise Exception(
                f"Activations directory ({store_cfg.cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )
    else:
        os.makedirs(store_cfg.cached_activations_path)

    print(f"Started caching {cfg.total_training_tokens} activations")
    tokens_per_buffer = (
        cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer
    )
    n_buffers = math.ceil(cfg.total_training_tokens / tokens_per_buffer)
    # for i in tqdm(range(n_buffers), desc="Caching activations"):
    for i in range(n_buffers):
        buffer = activations_store.get_buffer(cfg.n_batches_in_buffer)
        torch.save(buffer, f"{store_cfg.cached_activations_path}/{i}.pt")
        del buffer

        if i % cfg.shuffle_every_n_buffers == 0 and i > 0:
            # Shuffle the buffers on disk

            # Do random pairwise shuffling between the last shuffle_every_n_buffers buffers
            for _ in range(cfg.n_shuffles_with_last_section):
                shuffle_activations_pairwise(
                    store_cfg.cached_activations_path,
                    buffer_idx_range=(i - cfg.shuffle_every_n_buffers, i),
                )

            # Do more random pairwise shuffling between all the buffers
            for _ in range(cfg.n_shuffles_in_entire_dir):
                shuffle_activations_pairwise(
                    store_cfg.cached_activations_path,
                    buffer_idx_range=(0, i),
                )

    # More final shuffling (mostly in case we didn't end on an i divisible by shuffle_every_n_buffers)
    if n_buffers > 1:
        for _ in tqdm(range(cfg.n_shuffles_final), desc="Final shuffling"):
            shuffle_activations_pairwise(
                store_cfg.cached_activations_path,
                buffer_idx_range=(0, n_buffers),
            )
