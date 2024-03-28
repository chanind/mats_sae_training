import gzip
import os
import pickle
from dataclasses import dataclass, fields
from typing import Any, Iterable, Iterator

import torch

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sparse_autoencoder import SparseAutoencoder


@dataclass
class SAEGroupSharedConfig:
    """
    These portions of the config must not vary between SparseAutoencoder instances in a SAEGroup
    """

    model_name: str
    d_in: int
    log_to_wandb: bool
    wandb_project: str
    run_name: str | None
    train_batch_size: int
    feature_sampling_window: int
    wandb_log_frequency: int
    n_checkpoints: int
    checkpoint_path: str
    dataset_path: str
    cached_activations_path: str | None
    use_cached_activations: bool
    is_dataset_tokenized: bool
    total_training_tokens: int
    n_batches_in_buffer: int
    store_batch_size: int
    context_size: int
    device: str | torch.device
    hook_point_template: str
    hook_point_head_index: int | None
    dtype: torch.dtype
    prepend_bos: bool


class SAEGroup:
    autoencoders: list[SparseAutoencoder]

    def __init__(
        self,
        cfgs: LanguageModelSAERunnerConfig | Iterable[LanguageModelSAERunnerConfig],
    ):
        # for loading old SAEGroups, when cfgs was a LanguageModelSAERunnerConfig object
        if isinstance(cfgs, LanguageModelSAERunnerConfig):
            cfgs = [cfgs]
        _validate_cfgs(cfgs)
        self.autoencoders = [SparseAutoencoder(cfg) for cfg in cfgs]

    def __iter__(self) -> Iterator[SparseAutoencoder]:
        # Make SAEGroup iterable over its SparseAutoencoder instances and their parameters
        for ae in self.autoencoders:
            yield ae  # Yielding as a tuple

    def __len__(self):
        # Return the number of SparseAutoencoder instances
        return len(self.autoencoders)

    def to(self, device: torch.device | str):
        for ae in self.autoencoders:
            ae.to(device)

    @property
    def shared_config(self) -> SAEGroupSharedConfig:
        # we validate that all the autoencoders have identical values for these fields
        return SAEGroupSharedConfig(
            model_name=self.autoencoders[0].cfg.model_name,
            log_to_wandb=self.autoencoders[0].cfg.log_to_wandb,
            wandb_project=self.autoencoders[0].cfg.wandb_project,
            run_name=self.autoencoders[0].cfg.run_name,
            train_batch_size=self.autoencoders[0].cfg.train_batch_size,
            feature_sampling_window=self.autoencoders[0].cfg.feature_sampling_window,
            wandb_log_frequency=self.autoencoders[0].cfg.wandb_log_frequency,
            n_checkpoints=self.autoencoders[0].cfg.n_checkpoints,
            checkpoint_path=self.autoencoders[0].cfg.checkpoint_path,
            dataset_path=self.autoencoders[0].cfg.dataset_path,
            cached_activations_path=self.autoencoders[0].cfg.cached_activations_path,
            use_cached_activations=self.autoencoders[0].cfg.use_cached_activations,
            is_dataset_tokenized=self.autoencoders[0].cfg.is_dataset_tokenized,
            total_training_tokens=self.autoencoders[0].cfg.total_training_tokens,
            n_batches_in_buffer=self.autoencoders[0].cfg.n_batches_in_buffer,
            store_batch_size=self.autoencoders[0].cfg.store_batch_size,
            context_size=self.autoencoders[0].cfg.context_size,
            device=self.autoencoders[0].cfg.device,
            hook_point_template=self.autoencoders[0].cfg.hook_point_template,
            hook_point_head_index=self.autoencoders[0].cfg.hook_point_head_index,
            dtype=self.autoencoders[0].cfg.dtype,
            d_in=self.autoencoders[0].cfg.d_in,
            prepend_bos=self.autoencoders[0].cfg.prepend_bos,
        )

    def get_combined_sae_configs(self) -> dict[str, Any | list[Any]]:
        """
        Combines the configurations of all SparseAutoencoder instances in the SAEGroup.
        If a configuration value is the same for all instances, it is included as a single value.
        If a configuration value differs between instances, it is included as a list of values.
        """
        combined = {}
        for cfg in self.autoencoders:
            for key, value in cfg.__dict__.items():
                if key not in combined:
                    combined[key] = [value]
                elif value not in combined[key]:
                    combined[key].append(value)
        for key, value in combined.items():
            if len(value) == 1:
                combined[key] = value[0]
        return combined

    @classmethod
    def load_from_pretrained(cls, path: str) -> Any:
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    group = torch.load(path, map_location="mps")
                    if isinstance(group, dict):
                        group["cfg"].device = "mps"
                    else:
                        for sae in group.autoencoders:
                            sae.cfg.device = "mps"
                else:
                    group = torch.load(path)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    group = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    group = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        return group
        # # # Ensure the loaded state contains both 'cfg' and 'state_dict'
        # # if "cfg" not in state_dict or "state_dict" not in state_dict:
        # #     raise ValueError(
        # #         "The loaded state dictionary must contain 'cfg' and 'state_dict' keys"
        # #     )

        # # Create an instance of the class using the loaded configuration
        # instance = cls(cfg=state_dict["cfg"])
        # instance.load_state_dict(state_dict["state_dict"])

        # return instance

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        if path.endswith(".pt"):
            torch.save(self, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    def get_name(self):
        # list of set to dedupe the layers
        layers = list({sae.cfg.hook_point_layer for sae in self.autoencoders})
        hook_point_templates = list(
            {sae.cfg.hook_point_template for sae in self.autoencoders}
        )
        hook_points = list({sae.cfg.hook_point for sae in self.autoencoders})
        d_saes = ",".join(list({str(sae.cfg.d_sae) for sae in self.autoencoders}))
        model_names = ",".join(
            sorted(list({sae.cfg.model_name for sae in self.autoencoders}))
        )
        if len(layers) > 1:
            layer_string = f"{min(layers)-max(layers)}"
        else:
            layer_string = f"{layers[0]}"
        if len(hook_point_templates) > 1:
            hook_point_str = hook_point_templates[0].format(layer=layer_string)
        else:
            hook_point_str = ",".join(hook_points)
        sae_name = f"sae_group_{model_names}_{hook_point_str}_{d_saes}"
        return sae_name

    def eval(self):
        for ae in self.autoencoders:
            ae.eval()

    def train(self):
        for ae in self.autoencoders:
            ae.train()


def _validate_cfgs(cfgs: Iterable[LanguageModelSAERunnerConfig]):
    """
    Assert that any fields which cannot vary between SparseAutoencoder instances in a SAEGroup are the same.
    """
    singular_fields = [field.name for field in fields(SAEGroupSharedConfig)]
    for field in singular_fields:
        values = {getattr(cfg, field) for cfg in cfgs}
        if len(values) > 1:
            raise ValueError(
                f"All configurations in the SAEGroup must have the same value for {field}"
            )
