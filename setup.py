from setuptools import setup

__VERSION__ = "0.1.0"

setup(
    name="spectral_graph_conv",
    version=__VERSION__,
    packages=["spectral_graph_conv"],
    python_requires=">=3.10",
    install_requires=[
        "dacite",
        "einops",
        "hydra-core",
        "numpy",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm",
        "types-tqdm",
        "wandb",
    ],
    include_package_data=True,
)
