import argparse

import torch
from tqdm import tqdm
import hydra
from hydra import initialize, compose
from pytictac import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Export a torchscript model")
    parser.add_argument("-config", type=str, help="yaml config file name")
    parser.add_argument("-checkpoint", type=str, help="checkpoint file")
    args = parser.parse_args()
    return args


def get_data(batched_data, device):
    (
        imgs,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        target,
        aux,
        *_,
        pcd_new,
        gvom_new,
    ) = batched_data

    target_tensor = torch.tensor(target.shape).to(device)
    for v in pcd_new.values():
        v.to(device)

    return [
        imgs.to(device),
        rots.to(device),
        trans.to(device),
        intrins.to(device),
        post_rots.to(device),
        post_trans.to(device),
        target_tensor.to(device),
        pcd_new,
        # pcd_new,
        # aux.to(device),
    ]


def main():
    args = parse_args()

    initialize(config_path="../../cfg", job_name="Inference Test")
    cfg = compose(config_name=args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

    # Load Model config
    cfg_model = cfg.model.network
    # Instanstiate model
    model = hydra.utils.instantiate(cfg_model)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint)

    state_dict = {
        k[len("net.") :] if k.startswith("net.") else k: v
        for k, v in ckpt["state_dict"].items()
        if "net" in k
    }

    model.load_state_dict(state_dict)
    model.to(device)

    # # Get dataset and a sample data
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    model.eval()
    data_iter = iter(train_dataloader)

    try:
        print("Attempting to Load Model")
        loaded_model = torch.jit.load("traced_model.pt")
        frozen_model = torch.jit.optimize_for_inference(loaded_model)
    except:
        print("Tracing the model since unable to load model")
        data = next(data_iter)
        dummy = get_data(data, device)

        traced_model = torch.jit.trace(
            model,
            dummy,
        )
        print(traced_model)
        # Save the traced model to a file
        traced_model.save("traced_model.pt")
        loaded_model = torch.jit.load("traced_model.pt")
        frozen_model = torch.jit.optimize_for_inference(loaded_model)

    with torch.no_grad():
        for idx, data_new in enumerate(train_dataloader):
            data_new = get_data(data_new, device)
            with Timer("torchscript_model"):
                output_traced = frozen_model(
                    imgs=data_new[0],
                    rots=data_new[1],
                    trans=data_new[2],
                    intrins=data_new[3],
                    post_rots=data_new[4],
                    post_trans=data_new[5],
                    target_shape=data_new[6],
                    pcd_new=data_new[7],
                )

            with Timer("regular model"):
                output_normal = model(
                    imgs=data_new[0],
                    rots=data_new[1],
                    trans=data_new[2],
                    intrins=data_new[3],
                    post_rots=data_new[4],
                    post_trans=data_new[5],
                    target_shape=data_new[6],
                    pcd_new=data_new[7],
                )

            # print("Traced output: ", output_traced)
            # print("Normal output: ", output_normal)
            tolerance = 1e-3
            diff_mask = (
                torch.abs(output_traced[0, 1, ...] - output_normal[0, 1, ...])
                > tolerance
            )
            # Get the sum where the difference exceeds the tolerance
            indices_diff = torch.sum(diff_mask)
            print(indices_diff)


if __name__ == "__main__":
    main()
