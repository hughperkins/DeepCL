/*
void backPropWeightsGpu3( const float learningRate, float const*const errors, float *const weightChanges ) {
    // each workgroup is dimensioned to be big enough to loop over the usptream Board
    // round to nearest 32, which about fills an average compute unit (16 or 32)
    const int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    // then, once we have the workgroup size, well, first how many workgroups?
    // it is: number outplanes * number inplanes:
    const int numWorkgroups = upstreamNumPlanes * numPlanes;
    //multiply, to get globalsize:
    const int globalSize = numWorkgroups * workgroupsize;
    std::cout << " workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;
    // yay :-)

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    CLWrapper *imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
    CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
    CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );

    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernelBackPropWeights3
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl->getNextPower2( workgroupsize ) )

       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( workgroupsize );

    kernelBackPropWeights3->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();

//        // reduce on cpu for now :-)
//        // need to reduce over ...
//        for( int filterId = 0; filterId < numPlanes; filterId++ ) {
//            for( int filterPos = 0; filterPos < filterBoardSizeSquared; filterPos++ ) {
//                float sum = 0;
//                for( int 
////                for( int workgroupId = 0; workgroupId < numWorkgroups; workgroupId++ ) {
//                    sum += weightChangesReduceArea[  ];
//                }
//                weightChanges[ filterId * filterBoardSizeSquared + filterPos ] = sum;
//            }
//        }

//        cl->finish();

//        timer.timeCheck("backPropGpu");
    delete imagesWrapper;
    delete resultsWrapper;
    delete errorsWrapper;
    delete weightChangesWrapper;
}
*/

