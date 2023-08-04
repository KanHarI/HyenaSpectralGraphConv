## _This repository is not currently working_
This is an attempt to look at the limitations of spectral graph convolutions when applied to random graphs.
The default run is an overfit on a tiny graph, does not generalize to large graphs at all.

# Setup:
Create a new venv, then:
```bash
pip install -U pip setuptools wheel
pip install -U -r dev-requirements.txt
pip install -e .
./lint_and_test.sh
```

# Running:
```bash
python spectral_graph_conv/experiments/undirected_tree_parent_restoration.py 
```

# Configuring the run:
Modify `spectral_graph_conv/conf/undirected_tree_parent_reconstruction.yaml`.

The current config is a tiny CPU run that will overfit on tiny random trees with 
4 nodes where every node is selected randomly from an alphabet of size 3 
(the network learns 4 tokens to predict - these 3, and a root node token).
