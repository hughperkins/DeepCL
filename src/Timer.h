// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>

#if (_MSC_VER == 1500) // visual studio 2008
#define MSVC2008
#include <ctime>
#else
#include <chrono>
#endif

#include <string>

class Timer{
public:
    #ifdef MSVC2008
    double last;
    #else
//   double last;
    std::chrono::time_point<std::chrono::high_resolution_clock> last;
    #endif
   Timer() {
//      last = clock();
      last = getCount();        
   }

//#ifdef _WIN32
#ifdef MSVC2008
   double getCount() {
        time_t thistime;
        time(&thistime);
//	    struct std::timeval tm;
//	    gettimeofday( &tm, NULL );
//	    return (double)tm.tv_sec + (double)tm.tv_usec / 1000000.0;
        return (double)thistime;
   }
#else
   std::chrono::time_point<std::chrono::high_resolution_clock> getCount() {
        return std::chrono::high_resolution_clock::now();
   }
#endif

   void timeCheck(std::string label ) {
//        #ifdef _WIN32
    #ifdef MSVC2008
       double thistime = getCount();
       double timemilliseconds = ( thistime - last ) * 1000.0;
        #else
     std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds> ( change ).count() );
        #endif
      last = thistime;
      std::cout << label << " " << timemilliseconds << " ms" << std::endl;
   }

   double lap() {
//       #ifdef _WIN32
    #ifdef MSVC2008
       double thistime = getCount();
      double timemilliseconds = 1000 * ( thistime - last );
       #else
      std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds> ( change ).count() );
       #endif
      last = thistime;
      return timemilliseconds;
   }
};

