/*
void ConvolutionalLayer::backPropWeightsGpuWithScratch( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, CLWrapper*errorsWrapper, CLWrapper *weightChangesWrapper ) {
//        Timer timer;
    StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratch start, layer " + toString( layerIndex ) );
//        int globalSize = getWeightsSize();
    int workgroupsize = filterSizeSquared;
    int numWorkgroups = upstreamNumPlanes * numPlanes;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        std::cout << " backpropgpuwithscratch, globalsize " << globalSize << " workgroupsize " << workgroupsize << std::endl;

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
//    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
//        imagesWrapper->copyToDevice();
//    errorsWrapper->copyToDevice();
    kernelBackPropWeightsWithScratch
       ->in(learningMultiplier)
       ->in( batchSize )
       
        ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )

        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared );
    kernelBackPropWeightsWithScratch->run_1d(globalSize, workgroupsize);

    cl->finish();

//        timer.timeCheck("backPropGpu");
//    delete errorsWrapper;
//        delete weightChangesWrapper;
    StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratch end, layer " + toString( layerIndex ) );
}
*/

