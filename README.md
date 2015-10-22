# PMVS-GPU
This project modifies [PMVS](http://www.di.ens.fr/pmvs/) to use the GPU. So far it has only been tested on Ubuntu 14.04 with NVIDIA graphics cards.

## Requirements
#### OpenCL 1.2
Verify OpenCL is configured correctly by running the `clinfo` utility. It should find a GPU device and return lots of info. On Ubuntu, OpenCL appears to be broken when using NVIDIA Ubuntu packages. If OpenCL isn't working, try removing all NVIDIA Ubuntu packages and reinstall drivers using the NVIDIA binary installer downloaded from [http://www.geforce.com/drivers](http://www.geforce.com/drivers).

#### Ubuntu packages

```
sudo apt-get install libgsl0-dev libblas-dev libatlas-dev liblapack-dev opencl-headers libjpeg-dev
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
There is a more up-to-date branch of OpenDroneMap called python-port. When the input images have GPS metadata, the point matching step is much faster for large datasets. To use python-port, first clone the OpenDroneMap repository, then check out the branch using git.
```
git clone https://github.com/OpenDroneMap/OpenDroneMap.git
cd OpenDroneMap
git fetch
git checkout python-port
./install.sh
```

OpenDroneMap comes with the original version of pmvs2. To use pmvs-gpu instead, copy the binary to the OpenDroneMap bin directory.
```cp <PMVS-GPU dir>/program/main/pmvs2 <OpenDroneMap dir>/bin```

After installing, run OpenDroneMap by launching the `run.py` script from the directory that contains the input images. Note that if you're using the default branch (gh-pages), use `run.pl` instead.
