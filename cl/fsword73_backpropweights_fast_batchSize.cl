// Copyright // FSWORD73 AT HOTMAIL DOT COM
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// Original workgroup by  hughperkins at gmail in deepCL
// globalId: [outPlane][inputPlane][filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
//27     int workgroupsize = std::max(32, square(dim.filterSize) ); // no point in wasting cores... 
//28     int numWorkgroups = dim.inputPlanes * dim.numFilters; 
//29     int globalSize = workgroupsize * numWorkgroups; 
//30     globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize; 

#define 	gNumFilters 	16
#define 	gInputPlanes	8
#define   gFilterSize   14 
#define   gFilterSizeSquared (gFilterSize*gFilterSize)
#define   gOutputSize     14
#define   gInputSize      14 
#define   gMargin         0

#define   FIXED_WORKGROUPSIZE 64 
#define   FIXED_WORKGROUPSIZE_SHIFT 6

void  calc_backprop_floats_batchSize(
        global const float *gradOutput, 
				global const float *images, 
				float *thiswchange,
				#ifdef BIASED  
				float*  thisbiaschange,	
				#endif 
				const int batchSize,
				int globalId,	
				int localId				
 ) 
 {	
			  *thiswchange = 0;	      			  	 
	      
			  int globalIdOutput = (globalId >> FIXED_WORKGROUPSIZE_SHIFT);	    
	 
				
				int IntraFilterOffset =  globalIdOutput % gFilterSizeSquared;
				int filterRow = IntraFilterOffset / gFilterSize;
				int filterCol = IntraFilterOffset % gFilterSize;

				int filter2Id = globalIdOutput / gFilterSizeSquared;
				int outPlane = filter2Id / gInputPlanes;
				int upstreamPlane = filter2Id % gInputPlanes;	 				
	 
			  int iterations = (batchSize + FIXED_WORKGROUPSIZE -1) >> FIXED_WORKGROUPSIZE_SHIFT;
	 
	     for (int i = 0; i < iterations; i++) {
				int n = i * FIXED_WORKGROUPSIZE + localId;
				if( n >= batchSize)
					break;

				int upstreamRow = 0 - gMargin + filterRow;
				for (int outRow = 0; outRow < gOutputSize; outRow++,upstreamRow++) {
					
					bool proceed0 = upstreamRow >= 0 && upstreamRow < gInputSize;
					if(	proceed0 == false)
					{
						 continue;
					}
					
					
					int resultIndex = (( n * gNumFilters 
										+ outPlane) * gOutputSize
										+ outRow) * gOutputSize
					          + 0;
					
					int upstreamCol =  0 - gMargin + filterCol; 
					int upstreamDataIndex= (( n * gInputPlanes 
													 + upstreamPlane) * gInputSize
													 + upstreamRow) * gInputSize
													 + upstreamCol;					
					
					for (int outCol = 0; outCol < gOutputSize; outCol++,	upstreamCol++, upstreamDataIndex++, resultIndex++){
						//int upstreamCol = outCol - gMargin + filterCol;
						//bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize
						//		&& upstreamCol < gInputSize;
						
						bool proceed  = upstreamCol >=0 && upstreamCol < gInputSize && proceed0;					
						

						if (proceed) {
							//int resultIndex = (( n * gNumFilters 
							//					+ outPlane) * gOutputSize
							//					+ outRow) * gOutputSize
							//					+ outCol;
							//int resultIndex = resultIndex0 + outCol;
							float error = gradOutput[resultIndex];
							//int upstreamDataIndex = (( n * gInputPlanes 
							//								 + upstreamPlane) * gInputSize
							//								 + upstreamRow) * gInputSize
							//								 + upstreamCol;
							//int upstreamDataIndex = upstreamDataIndex0 + upstreamCol;
							float upstreamResult = images[upstreamDataIndex];
							float thisimagethiswchange = upstreamResult * error;
							*thiswchange += thisimagethiswchange;
							#ifdef BIASED
							*thisbiaschange += error;
							#endif
						}
			
					}
					
				}
			}		

 } 
 
 void Reduction_of_Weights( 
				float* thiswchange,
				__local  float* sdata, 				
				#ifdef BIASED  
				 float*  thisbiaschange,	
				#endif			  
				int globalId,
				int localId)
 {
		  
	//store into local 	 
	sdata[localId] = *thiswchange;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(unsigned int s = FIXED_WORKGROUPSIZE >>1; s > 0; s >>= 1) 
	{
		if(localId < s) 
		{
			sdata[localId] += sdata[localId + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(localId == 0)
	{
		*thiswchange = 	sdata[0];
		
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#ifdef BIASED  
	sdata[localId] = *thisbiaschange;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(unsigned int s = FIXED_WORKGROUPSIZE >>1; s > 0; s >>= 1) 
	{
		if(localId < s) 
		{
			sdata[localId] += sdata[localId + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
		
	if(localId == 0)
	{
		*thisbiaschange = 	sdata[0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif
}
			
			
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, 
                             prevVal.intVal, newVal.intVal) 
                             != prevVal.intVal);
}

//   
//   Global ID :  [outPlane][inputPlane][filterRow][filterCol] *  BatchSize * 64   
//   Per thread = ( gOutputSize * gOutputSize)/64
//   workgroupsize =  64 treads == 1 image  
//   numWorkgroups  = [outPlane][inputPlane][filterRow][filterCol] *  BatchSize 
//   int globalSize = workgroupsize * numWorkgroups; 
//   Assuming   [outPlane][inputPlane][filterRow][filterCol] is cleared or preset 
//   Each workgroupsize call Atomic_Add 
//	  One pass to compete it. 
//    Stage1:  caclaute each pixels 
//    Stage2:  Rediction whole workgroupsize into 1 
//    Local LDS = 64;

#if 1 
void __kernel test_kernel(				
						const int batchSize,  
						const float learningRateMultiplier,
						__global float* images,	
//						__global const float* filter,
						__global const float* gradOutput,
						__global float* gradWeights
             #ifdef BIASED
                     ,__global float *gradBiasWeights
             #endif						 
 )

#else 
void __kernel backprop_floats_fast(				
        __global const float *gradOutput, 
				__global const float *images, 
        __global float *gradWeights,				
        #ifdef BIASED
             __global float *gradBiasWeights,
        #endif
				const float learningRateMultiplier,
				const int batchSize,
			  const int gOutputSize,  	
 ) 
 #endif
 {	 
	 
//	 const float learningRateMultiplier = 0.0001f;
	 int globalId = get_global_id(0);				
	 int localId  = get_local_id(0);
	  float thiswchange = 0;
	  __local float sdata[FIXED_WORKGROUPSIZE];	
#ifdef BIASED
	 float thisbiaschange  = 0;
#endif
	  
	 //It does not include any Invalid Threads since the FIXED_WORKGROUPSIZE is 64)
   // if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize *  batchSize * FIXED_WORKGROUPSIZE) {
   //     //Do nothing	 
   // }
	 
	 calc_backprop_floats_batchSize(
        gradOutput, 
				images, 
				&thiswchange,
				#ifdef BIASED  
				&thisbiaschange,	
				#endif 
				batchSize,
				globalId,	
				localId);	


	//aggregate Data to Thread0
	Reduction_of_Weights( 
				&thiswchange,
				sdata,			
#ifdef BIASED  
				&thisbiaschange,	
#endif				
				globalId,
				localId
	);
		
	if(localId == 0)
	{
		int globalIdOutput = (globalId >> FIXED_WORKGROUPSIZE_SHIFT);
		gradWeights[ globalIdOutput ] = thiswchange;
		
#ifdef BIASED  
		int filter2Id = globalIdOutput / gFilterSizeSquared;
		int outPlane = filter2Id / gInputPlanes;
		gradBiasWeights[outPlane] = thisbiaschange;
#endif																		
	}						

}

//CNNBench command line 
//-dim 1 -lx 64 -ly 1 -gx 2359296 -gy 1 -f 3 -c1 128 -c2 128 -i 1 -x 4096 -y 4096
	//    Input 	128x128, 4 Planes, BatchSize 128 
  //    output  128x128, 8 planes,  BatchSize 128    
  //    Filter Size = 3x3 
	//   numWorkgroups  = [outPlane][inputPlane][filterRow][filterCol] *  BatchSize 
  //   int globalSize = workgroupsize * numWorkgroups; 
	//    globalSize = 64 * [8][4][3][3] *  128   = 2359296
  //     
	//    	    Input 	= 128x128x4x128 batch = 4096 * 2048
  //			    output  = 128x128x8x128 batch = 4096 * 4096  
	//          force localthread_x =64,  globalthread_y =1
	//          FilterSize =3,  
	//         -constant1 = batchSize 128
	//         -constant2 = gOutputSize 128
