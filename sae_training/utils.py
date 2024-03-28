from collections.abc import Iterable
from typing import NamedTuple, Tuple

import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore, ActivationsStoreConfig
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sae_group import SAEGroup
from sae_training.sparse_autoencoder import SparseAutoencoder


class SAEGroupSession(NamedTuple):
    model: HookedTransformer
    sae_group: SAEGroup
    activations_loader: ActivationsStore


class LMSparseAutoencoderSessionloader:
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    cfgs: LanguageModelSAERunnerConfig | Iterable[LanguageModelSAERunnerConfig]

    def __init__(
        self,
        cfgs: LanguageModelSAERunnerConfig | Iterable[LanguageModelSAERunnerConfig],
    ):
        self.cfgs = cfgs

    def load_session(
        self,
    ) -> SAEGroupSession:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """
        sae_group = SAEGroup(self.cfgs)
        shared_config = sae_group.shared_config
        model = self.get_model(shared_config.model_name)
        model.to(shared_config.device)
        activations_loader = self.get_activations_store(sae_group, model)

        return SAEGroupSession(model, sae_group, activations_loader)

    @classmethod
    def load_session_from_pretrained(cls, path: str) -> SAEGroupSession:
        """
        Loads a session for analysing a pretrained sparse autoencoder group.
        """
        # if torch.backends.mps.is_available():
        #     cfg = torch.load(path, map_location="mps")["cfg"]
        #     cfg.device = "mps"
        # elif torch.cuda.is_available():
        #     cfg = torch.load(path, map_location="cuda")["cfg"]
        # else:
        #     cfg = torch.load(path, map_location="cpu")["cfg"]

        sparse_autoencoders = SAEGroup.load_from_pretrained(path)

        # hacky code to deal with old SAE saves
        if type(sparse_autoencoders) is dict:
            sparse_autoencoder = SparseAutoencoder(cfg=sparse_autoencoders["cfg"])
            sparse_autoencoder.load_state_dict(sparse_autoencoders["state_dict"])
            model, sparse_autoencoders, activations_loader = cls(
                sparse_autoencoder.cfg
            ).load_session()
            sparse_autoencoders.autoencoders[0] = sparse_autoencoder
        elif type(sparse_autoencoders) is SAEGroup:
            cfgs = [sae.cfg for sae in sparse_autoencoders.autoencoders]
            model, _, activations_loader = cls(cfgs).load_session()
        else:
            raise ValueError(
                "The loaded sparse_autoencoders object is neither an SAE dict nor a SAEGroup"
            )

        return SAEGroupSession(model, sparse_autoencoders, activations_loader)

    def get_model(self, model_name: str):
        """
        Loads a model from transformer lens
        """

        # Todo: add check that model_name is valid

        model = HookedTransformer.from_pretrained(model_name)

        return model

    def get_activations_store(
        self,
        sae_group: SAEGroup,
        model: HookedTransformer,
    ):
        """
        Loads a DataLoaderBuffer for the activations of a language model.
        """
        shared_cfg = sae_group.shared_config
        layers = sorted(
            list({sae.cfg.hook_point_layer for sae in sae_group.autoencoders})
        )
        activations_store_cfg = ActivationsStoreConfig(
            hook_point_template=shared_cfg.hook_point_template,
            hook_point_head_index=shared_cfg.hook_point_head_index,
            dataset_path=shared_cfg.dataset_path,
            is_dataset_tokenized=shared_cfg.is_dataset_tokenized,
            context_size=shared_cfg.context_size,
            use_cached_activations=shared_cfg.use_cached_activations,
            cached_activations_path=shared_cfg.cached_activations_path,
            n_batches_in_buffer=shared_cfg.n_batches_in_buffer,
            total_training_tokens=shared_cfg.total_training_tokens,
            store_batch_size=shared_cfg.store_batch_size,
            d_in=shared_cfg.d_in,
            device=shared_cfg.device,
            dtype=shared_cfg.dtype,
            train_batch_size=shared_cfg.train_batch_size,
            hook_point_layers=layers,
            prepend_bos=shared_cfg.prepend_bos,
        )

        activations_loader = ActivationsStore(
            activations_store_cfg,
            model,
        )

        return activations_loader


def shuffle_activations_pairwise(datapath: str, buffer_idx_range: Tuple[int, int]):
    """
    Shuffles two buffers on disk.
    """
    assert (
        buffer_idx_range[0] < buffer_idx_range[1] - 1
    ), "buffer_idx_range[0] must be smaller than buffer_idx_range[1] by at least 1"

    buffer_idx1 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    buffer_idx2 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    while buffer_idx1 == buffer_idx2:  # Make sure they're not the same
        buffer_idx2 = torch.randint(
            buffer_idx_range[0], buffer_idx_range[1], (1,)
        ).item()

    buffer1 = torch.load(f"{datapath}/{buffer_idx1}.pt")
    buffer2 = torch.load(f"{datapath}/{buffer_idx2}.pt")
    joint_buffer = torch.cat([buffer1, buffer2])

    # Shuffle them
    joint_buffer = joint_buffer[torch.randperm(joint_buffer.shape[0])]
    shuffled_buffer1 = joint_buffer[: buffer1.shape[0]]
    shuffled_buffer2 = joint_buffer[buffer1.shape[0] :]

    # Save them back
    torch.save(shuffled_buffer1, f"{datapath}/{buffer_idx1}.pt")
    torch.save(shuffled_buffer2, f"{datapath}/{buffer_idx2}.pt")
