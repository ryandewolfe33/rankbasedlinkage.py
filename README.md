# rankbasedlinkage.py

This is a pure python implementation of the rank based linkage clustering algorithm from the papser *Rank-based linkage i: triplet comparisons and oriented simplicial complexes* by R. W. R. Darling, W. Grilliette, and A. Logan.
The paper comes with an [official Java implementation](https://github.com/NationalSecurityAgency/rank-based-linkage) that is much faster.

The algorithm is implemented with an approximately sci-kit learn class, so the usage should be familar to those with sci-kit experience.
Check out the notebooks for example usage, or getting started below.

## Getting Started
For now the package must be cloned and locally installed.
```bash
git clone git@github.com:ryandewolfe33/rankbasedlinkage.py.git
cd rankbasedlinkage.py
pip install .
```

## Example Usage
```python
import sklearn.datasets as data
import rankbasedlinkage as rbl
moons, _ = data.make_moons(n_samples=50, noise=0.05, random_state=123)
rank_graph = rbl.build_knn_rank_graph(moons)
rbl_labels = rbl.RankBasedLinkage.fit_predict(rank_graph)
```
