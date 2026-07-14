"""
Exports the trained model to TorchScript.
Run:
    python -m src.export_model
"""
import torch

from src import config
from src.model import AttentionUNet


def main():
    model = AttentionUNet(img_size=config.IMG_SIZE, pretrained=False)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location="cpu"))
    model.eval()

    example_input = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    traced_model = torch.jit.trace(model, example_input)

    # Quick sanity check: traced model output must match the eager model's output
    with torch.no_grad():
        eager_out = model(example_input)
        traced_out = traced_model(example_input)
        max_diff = (eager_out - traced_out).abs().max().item()
    assert max_diff < 1e-5, f"Traced model diverges from eager model! max_diff={max_diff}"

    traced_model.save(str(config.TORCHSCRIPT_MODEL_PATH))
    print(f"Saved TorchScript model to {config.TORCHSCRIPT_MODEL_PATH} (max_diff vs eager: {max_diff:.2e})")


if __name__ == "__main__":
    main()
