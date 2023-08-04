black spectral_graph_conv
isort --profile black spectral_graph_conv
pip install -e .
flake8 spectral_graph_conv
mypy --strict spectral_graph_conv
pytest ./spectral_graph_conv/tests