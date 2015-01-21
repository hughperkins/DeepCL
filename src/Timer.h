// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
//#include <ctime>
//#include <sys/time.h>
#include <chrono>

class Timer{
public:
   // clock_t last;
//   double last;
    std::chrono::time_point<std::chrono::high_resolution_clock> last;
   Timer() {
      //last = clock();
      last = getCount();        
   }

   std::chrono::time_point<std::chrono::high_resolution_clock> getCount() {
        return std::chrono::high_resolution_clock::now();
//	    struct timeval tm;
//	    gettimeofday( &tm, NULL );
//	    return (double)tm.tv_sec + (double)tm.tv_usec / 1000000.0;
   }

   void timeCheck(std::string label ) {
//      double thistime = getCount();
     std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = std::chrono::duration_cast<std::chrono::milliseconds> ( change ).count();
      last = thistime;
      std::cout << label << " " << timemilliseconds << " ms" << std::endl;
   }

   double lap() {
      std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = std::chrono::duration_cast<std::chrono::milliseconds> ( change ).count();
//      double timemilliseconds = 1000 * ( change );
      last = thistime;
      return timemilliseconds;
   }
};

