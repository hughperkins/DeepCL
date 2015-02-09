
kernel void setupBoards( global float *boards, const int numBoards, const int boardSize ) {
    int globalId = get_global_id(0);
    if( globalId > numBoards * boardSize * boardSize ) {
        return;
    }
    boards[globalId] = ( globalId % 231 ) / 231.0f;
}

kernel void sum_threadperboard( global const float *boards, global float *sums, const int numBoards, const int boardSize ) {
    int globalId = get_global_id(0);
    if( globalId > numBoards ) {
        return;
    }
    float sum = 0;
    const int boardSizeSquared = boardSize * boardSize;
    global float const*thisBoard = boards + boardSizeSquared * globalId;
    for( int i = 0; i < boardSizeSquared; i++ ) {
        sum += thisBoard[i];
    }
    sums[globalId] = sum;
}

kernel void sumSums_singlethread( global const float *sums, global float *sum, const int numBoards ) {
    int globalId = get_global_id(0);
    if( globalId > 0 ) {
        return;
    }
    float thissum = 0;
    for( int i = 0; i < numBoards; i++ ) {
        thissum += sums[i];
    }
    sum[0] = thissum;
}

kernel void sum_sumrow( global const float *boards, global float *sums, const int numRows, const int rowSize ) {
    int globalId = get_global_id(0);
    if( globalId > numRows ) {
        return;
    }
    float sum = 0;
//    const int boardSizeSquared = boardSize * boardSize;
    global float const*thisrow = boards + rowSize * globalId;
    for( int i = 0; i < rowSize; i++ ) {
        sum += thisrow[i];
    }
    sums[globalId] = sum;
}

