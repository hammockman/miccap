# miccap

Data acquisition using an NI USB-9234.

The 9234 does not support hardware triggering https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019QljSAE&l=en-NZ.


Targeting a python environment:

```sh
> conda create --name ni python=3.9 scipy h5py
> conda install -c conda-forge nidaqmx-python
> conda install -c conda-forge wxpython
> conda install -c conda-forge matplotlib
```