import argparse
from pathlib import Path

import torch
from clika_ace import ClikaModule


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA RetinaFace Model")
    parser.add_argument("path_to_ckpt", type=Path, help="Path to saved checkpoint")
    parser.add_argument("--output_dir", type=Path, default=None, help="Path to save deployed model.")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.path_to_ckpt.parent
    return args


def main():
    args = parse_args()
    state_dict = torch.load(args.path_to_ckpt)
    if isinstance(state_dict, bytes):
        clika_module = ClikaModule.clika_from_serialized(state_dict)
    else:
        clika_module = ClikaModule()
        clika_module.load_state_dict(state_dict["state_dict"])
    f: str = args.output_dir.joinpath(args.path_to_ckpt.stem + ".onnx")
    dummy_inputs: dict = clika_module.clika_generate_dummy_inputs()
    torch.onnx.export(
        model=clika_module, f=f, args=dummy_inputs, input_names=["x"], dynamic_axes={"x": {0: "batch_size"}}
    )


if __name__ == "__main__":
    main()
