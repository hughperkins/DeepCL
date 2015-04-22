# Benchmarking

## Soumith benchmarks

### Results on Titan

Soumith benchmarks now include DeepCL, [Soumith benchmarks](https://github.com/soumith/convnet-benchmarks)
* For Go-board sized images, DeepCL is only about twice as slow
* For larger images, DeepCL is even slower :-P
* Anyway, now we have some benchmarks, we can gradually improve this

### Results on K520

I've created a Jenkins job to run the Soumith benchmarks on a K520.  I will probably
modify it to automatically upload the results to 
[K520 results](http://hughperkins.github.io/DeepCL/benchmarking/index.html)

## Other benchmarks

* Personally, I've been targeting Go-boards for the [kgsgov2 dataset](https://github.com/hughperkins/kgsgo-dataset-preprocessor), but I might just concentrate on the Soumith benchmarks for a bit.  Or I might make a benchmark test for Go-board sized images.  They're kind of implicit in the Soumith benchmarks in layer 4 though, and we don't really have any data for other implementations, so maybe best to just concentrate on the Soumith benchmarks?

