# miccap

Data acquisition using an NI USB-9234.

The 9234 does not support hardware triggering https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019QljSAE&l=en-NZ.


Targeting a python environment:

```sh
> conda create --name ni -c conda-forge python=3.9 scipy h5py nidaqmx-python wxpython matplotlib
```