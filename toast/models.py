"""Build a ToaSt-compressed model from a timm backbone + a YAML config."""
import yaml
import timm

from .scwp import apply_scwp
from .tcs import apply_tcs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_toast_model(config, pretrained=True, num_classes=1000,
                      scwp=True, tcs=True):
    """Create timm model, apply SCWP (offline) and TCS (inference-time).

    config keys: timm_name, mhsa_ratio, mhsa_skip, fc1_ratios, fc2_ratios,
                 use_attn, sample_lo, sample_hi.
    Set scwp=False to load an already-pruned+finetuned checkpoint separately.
    """
    model = timm.create_model(config["timm_name"], pretrained=pretrained,
                              num_classes=num_classes)
    if scwp:
        apply_scwp(model, config["mhsa_ratio"], tuple(config.get("mhsa_skip", (0,))))
    if tcs:
        apply_tcs(model, config["fc1_ratios"], config["fc2_ratios"],
                  use_attn=config.get("use_attn", False),
                  sample_lo=config.get("sample_lo", 0.02),
                  sample_hi=config.get("sample_hi", 0.20))
    return model
