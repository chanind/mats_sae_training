from dataclasses import asdict

import torch

from sae_training.config import LanguageModelSAERunnerConfig
from tests.unit.helpers import FIXTURES_DIR


def test_LanguageModelSAERunnerConfig_can_load_old_config():
    old_config_path = FIXTURES_DIR / "cfg_blocks.5.hook_resid_pre.pt"
    cfg = torch.load(old_config_path)
    assert isinstance(cfg, LanguageModelSAERunnerConfig)
    assert cfg.model_name == "gpt2-small"
    assert cfg.hook_point_template == "blocks.{layer}.hook_resid_pre"
    assert cfg.hook_point_layer == 5
    assert cfg.hook_point == "blocks.5.hook_resid_pre"
    assert asdict(cfg) == {
        "b_dec_init_method": "geometric_median",
        "cached_activations_path": "activations/Skylion007_openwebtext/gpt2-small/blocks.5.hook_resid_pre",
        "checkpoint_path": "checkpoints/65ufbyeo",
        "context_size": 128,
        "d_in": 768,
        "d_sae": 24576,
        "dataset_path": "Skylion007/openwebtext",
        "dead_feature_threshold": 1e-08,
        "dead_feature_window": 5000,
        "device": torch.device("cuda"),
        "dtype": torch.float32,
        "expansion_factor": 32,
        "feature_sampling_window": 1000,
        "from_pretrained_path": None,
        "hook_point_head_index": None,
        "hook_point_layer": 5,
        "hook_point_template": "blocks.{layer}.hook_resid_pre",
        "is_dataset_tokenized": False,
        "l1_coefficient": 8e-05,
        "log_to_wandb": True,
        "lp_norm": 1,
        "lr": 0.0004,
        "lr_scheduler_name": None,
        "lr_warm_up_steps": 5000,
        "model_name": "gpt2-small",
        "n_batches_in_buffer": 128,
        "n_checkpoints": 10,
        "prepend_bos": True,
        "run_name": "24576-L1-8e-05-LR-0.0004-Tokens-3.000e+08",
        "seed": 42,
        "store_batch_size": 32,
        "total_training_tokens": 300000000,
        "train_batch_size": 4096,
        "use_cached_activations": False,
        "use_ghost_grads": True,
        "verbose": True,
        "wandb_entity": None,
        "wandb_log_frequency": 100,
        "wandb_project": "mats_sae_training_gpt2_small_resid_pre_5",
    }
