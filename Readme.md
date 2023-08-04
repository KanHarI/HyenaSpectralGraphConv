# An attempt to adapt the Hyena paper to spectral graph convolutions on small random graphs

_Note: Not recommended for real world usages. It seems that spectral graph convolutions work well for 
tasks requiring finding regularities in large graphs, not for tasks requiring high 
resolution understanding of a single graph._

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
Modify `./spectral_graph_conv/conf/undirected_tree_parent_reconstruction.yaml`.

