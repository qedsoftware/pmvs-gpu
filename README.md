# PMVS-GPU
This project modifies [PMVS](http://www.di.ens.fr/pmvs/) to use the GPU.

## Requirements
#### OpenCL 1.2
Verify OpenCL is configured correctly by running the `clinfo` utility. It should find a GPU device and return lots of info. On Ubuntu, OpenCL appears to be broken when using NVIDIA Ubuntu packages. If OpenCL isn't working, try removing all NVIDIA Ubuntu packages and reinstall drivers using the NVIDIA binary installer downloaded from [http://www.geforce.com/drivers](http://www.geforce.com/drivers).

#### Ubuntu packages
```
sudo apt-get install libgsl0-dev libblas-dev libatlas-dev liblapack-dev liblapacke-dev
```

#### Other
* [Graclus](http://www.cs.utexas.edu/users/dml/Software/graclus.html)

## Build Instructions
Update `program/main/Makefile` to point to your graclus dir (`YOUR_INCLUDE_METIS_PATH` and `YOUR_LDLIB_PATH`)
```
cd program/main
make
sudo make install
```

## Using with OpenDroneMap
After running OpenDroneMap install script:
`cp program/main/pmvs2 <OpenDroneMap dir>/bin`
