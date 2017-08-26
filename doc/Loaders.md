# Loaders

This page documents available file format loaders.


## mnist

* format details: [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
* simply specify the path to the mnist images file, and the mnist labels file will be located automatically, based on the name

## norb

* format details: [NORB-small dataset](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
* simply specify the path to the norb images file, and the norb labels file will be located automatically, based on the name

## kgsv2

* format details: [https://github.com/hughperkins/kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
* simply specify the path to the kgsv2 .dat file, eg `trainkgsv2.dat`

## jpegs

(New, in 5.8.0!)

* this format comprises:
  * jpeg images
  * and a single manifest text file
* jpeg images should obey certain properties:
  * be uniformally sized
  * should not have spaces in the filename, or in the directory path
* RGB and greyscale are both ok: simply change `planes=` parameter to `planes=1`(greyscale) or `planes=3`(RGB)
* manifest format looks like this:
```
# format=deepcl-jpeg-list-v1 planes=1 width=28 height=28
/norep/data/mnist/imagenet/R1313411/0.JPEG 5
/norep/data/mnist/imagenet/R1316044/1.JPEG 0
/norep/data/mnist/imagenet/R1311530/2.JPEG 4
/norep/data/mnist/imagenet/R1315845/3.JPEG 1
/norep/data/mnist/imagenet/R1316670/4.JPEG 9
/norep/data/mnist/imagenet/R1313848/5.JPEG 2
/norep/data/mnist/imagenet/R1315845/6.JPEG 1
... etc ...
```
* ie, top line is a header line, stating the name of the format, and the dimensions of the data set
  * RGB, use `planes=3`, greyscale images use `planes=1`
* other lines all have one filepath, a single space, and the category label
  * category label is integer, zero-based
* Simply pass in the name of the manifest file to deepcl commandline, and deepcl will handle the rest, eg:
```bash
./deepcl_train datadir=/my/data/dir trainfile=train-manifest.txt validatefile=validate-manifest.txt
```
* You can create a simple test dataset from mnist dataset, to reassure yourself this work, as follows:
```bash
./mnist-to-jpegs /my/data/dir/mnist/train-images-idx3-ubyte /my/data/dir/mnist/imagenet 1280
# train:
./deepcl_train datadir=/my/data/dir/mnist/imagenet trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280
# yes, this uses the same data file for validation and training, but it's just to show the format works, not to rigorously
# test our mnist validation accuracy ;-)
```


