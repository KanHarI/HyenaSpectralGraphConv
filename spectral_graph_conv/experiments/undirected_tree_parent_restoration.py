import os
from typing import Any, Iterator

import dacite
import hydra
import torch
import torch.utils.data
import wandb

from spectral_graph_conv.conf.undirected_tree_parent_reconstruction import (
    UndirectedTreeParentReconstructionConf,
)
from spectral_graph_conv.dataset.random_tree_dataset import (
    RandomTreeDatasetConfig,
    RandomTreeSpectralDataset,
)
from spectral_graph_conv.models.toy_graph_spectral_resnet import (
    ToyGraphSpectralResnet,
    ToyGraphSpectralResnetConfig,
)
from spectral_graph_conv.utils.loop_dataloader import loop_dataloader


@hydra.main(
    config_path="../conf",
    config_name="undirected_tree_parent_reconstruction",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: UndirectedTreeParentReconstructionConf = dacite.from_dict(
        data_class=UndirectedTreeParentReconstructionConf, data=hydra_cfg
    )
    if config.experiment.wandb_log:
        wandb.init(
            project=config.experiment.project_name,
            name=config.experiment.run_name,
        )
        wandb.config.update(config)
    dataset_config = RandomTreeDatasetConfig(
        n_nodes=config.dataset.n_nodes,
        vocab_size=config.dataset.vocab_size,
        dtype=config.dtype_conf.dtype,
    )
    dataset = RandomTreeSpectralDataset(dataset_config)

    # We use random data, so we can use a single dataloader for training and validation.
    # Cant over-fit random data with respect to random data from the same distribution by definition
    dataloader: Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = loop_dataloader(
        torch.utils.data.DataLoader(
            dataset=dataset,  # type: ignore
            batch_size=config.optimizer.batch_size,
            num_workers=0,  # We are not performance intensive, allows easy porting to non-Unix
            pin_memory=True,
        )
    )
    toy_spectral_resnet_config = ToyGraphSpectralResnetConfig(
        n_layers=config.resnet.n_layers,
        filter_approximation_rank=config.resnet.filter_approximation_rank,
        dtype=config.dtype_conf.dtype,
        device=config.device,
        init_std=config.optimizer.init_std,
        n_embed=config.resnet.n_embed,
        linear_size_multiplier=config.resnet.linear_size_multiplier,
        activation=config.activation_conf.activation,
        dropout=config.resnet.dropout,
        ln_eps=config.resnet.ln_eps,
        vocab_size=config.dataset.vocab_size + 1,  # +1 for root node
        nll_epsilon=config.resnet.nll_epsilon,
    )
    toy_spectral_resnet = ToyGraphSpectralResnet(toy_spectral_resnet_config)
    toy_spectral_resnet.init_weights()
    optimizer = config.optimizer.create_optimizer(toy_spectral_resnet.parameters())
    toy_spectral_resnet.train()
    train_losses = torch.zeros(
        (config.experiment.eval_interval,), dtype=torch.float32, device="cpu"
    )
    train_losses += float("inf")  # Do not log first train loss as 0
    assert (
        config.experiment.eval_interval % config.optimizer.grad_accumulation_steps == 0
    )
    best_eval_loss = float("inf")
    for step in range(config.optimizer.max_iters):
        # LR schedule
        for param in optimizer.param_groups:
            param["lr"] = config.optimizer.get_lr(step)
        # Evaluate if needed
        if step % config.experiment.eval_interval == 0:
            with torch.no_grad():
                toy_spectral_resnet.eval()
                eval_losses = torch.zeros(
                    (config.experiment.eval_iters,), dtype=torch.float32, device="cpu"
                )
                for eval_step in range(config.experiment.eval_iters):
                    (
                        nodes,
                        parent_nodes,
                        _,  # We do not use the adjacency matrix directly
                        eigenvalues,
                        eigenvectors,
                        inv_eigenvectors,
                    ) = map(lambda x: x.to(config.device), next(dataloader))
                    loss = toy_spectral_resnet(
                        nodes, parent_nodes, eigenvalues, eigenvectors, inv_eigenvectors
                    )
                    eval_losses[eval_step] = loss.item()
                eval_loss = eval_losses.mean().item()
                train_loss = train_losses.mean().item()
                if config.experiment.wandb_log:
                    wandb.log(
                        {
                            "val/loss": eval_loss,
                            "lr": config.optimizer.get_lr(step),
                            "train/loss": train_loss,
                        },
                        step=step,
                    )
                print(
                    f"Step: {step}, train_loss: {train_loss}, eval_loss: {eval_loss}, lr: {config.optimizer.get_lr(step)}"
                )
                if eval_loss < best_eval_loss:
                    print(
                        f"Saving model to saved_models/{config.experiment.run_name}_best.pt"
                    )
                    best_eval_loss = eval_loss
                    if not os.path.exists("saved_models"):
                        os.makedirs("saved_models")
                    torch.save(
                        toy_spectral_resnet.state_dict(),
                        os.path.join(
                            "saved_models",
                            f"{config.experiment.run_name}_best.pt",
                        ),
                    )
            toy_spectral_resnet.train()
        # Train
        (
            nodes,
            parent_nodes,
            _,  # We do not use the adjacency matrix directly
            eigenvalues,
            eigenvectors,
            inv_eigenvectors,
        ) = map(lambda x: x.to(config.device), next(dataloader))
        loss = toy_spectral_resnet(
            nodes, parent_nodes, eigenvalues, eigenvectors, inv_eigenvectors
        )
        train_losses[step % config.experiment.eval_interval] = loss.item()
        if step % config.experiment.log_interval == 0:
            print(f"Step: {step}, loss: {loss.item()}")
        (loss / config.optimizer.grad_accumulation_steps).backward()
        if (step + 1) % config.optimizer.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    return 0


if __name__ == "__main__":
    main()
