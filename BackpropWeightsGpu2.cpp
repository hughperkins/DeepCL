/*
void backPropWeightsGpu2( float learningRate, float const*errors, float *weightChanges ) {
    // soooo.... going to feed in same data as before, but structure workgroups differently...

//        void kernel backprop_floats_2( const float learningRateMultiplier,
//        const int batchSize, const int upstreamNumPlanes, const int numOutPlanes, 
//         const int upstreamBoardSize, const int filterSize, const int outBoardSize, const int padZeros, 
//         global const float *upstreamBoardsGlobal, 
//         global const float *resultsGlobal, global const float *errorsGlobal,
//         global float *weightChangesGlobal ) {

    int globalSize = getWeightsSize();
//        int workgroupsize = cl->getMaxWorkgroupSize();
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    std::cout << " workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    CLWrapper *imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
    CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
    CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernelBackPropWeights2
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl->getNextPower2( workgroupsize ) )
//->in( upstreamNumPlanes )->in(numPlanes)
//           ->in( upstreamBoardSize )->in( filterSize )->in( boardSize )->in( padZeros ? 1 : 0 )

       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( workgroupsize );

    kernelBackPropWeights2->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();
//        cl->finish();

//        timer.timeCheck("backPropGpu");
    delete imagesWrapper;
    delete resultsWrapper;
    delete errorsWrapper;
    delete weightChangesWrapper;
}
*/

