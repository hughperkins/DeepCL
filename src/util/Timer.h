// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>

//#if (_MSC_VER == 1500 || _MSC_VER == 1600) // visual studio 2008 or 2010
#ifdef _MSC_VER // make consistent across all msvc versions, so dont have to retest on different msvc versions...
#define WINNOCHRONO 
#define NOMINMAX // prevents errors compiling std::max and std::min
#include <windows.h>
//#include <ctime>
#else
#include <chrono>
#endif

#include <string>

#include "DeepCLDllExport.h"

class DeepCL_EXPORT Timer{
public:
    #ifdef WINNOCHRONO
    DWORD last;
    #else
//   double last;
    std::chrono::time_point<std::chrono::high_resolution_clock> last;
    #endif
   Timer() {
//      last = clock();
      last = getCount();        
   }

//#ifdef _WIN32
#ifdef WINNOCHRONO
   DWORD getCount() {
      //  time_t thistime;
       // time(&thistime);
//	    struct std::timeval tm;
//	    gettimeofday(&tm, NULL);
//	    return (double)tm.tv_sec + (double)tm.tv_usec / 1000000.0;
        return timeGetTime();
   }
#else
   std::chrono::time_point<std::chrono::high_resolution_clock> getCount() {
        return std::chrono::high_resolution_clock::now();
   }
#endif

   void timeCheck(std::string label) {
//        #ifdef _WIN32
    #ifdef WINNOCHRONO
       DWORD thistime = getCount();
       DWORD timemilliseconds = (thistime - last);
        #else
     std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds> (change).count());
        #endif
      last = thistime;
      std::cout << label << " " << timemilliseconds << " ms" << std::endl;
   }

    double interval() { // gets interval since last 'lap' or 'timecheck', 
                        // without updating 'last'
    #ifdef WINNOCHRONO
       DWORD thistime = getCount();
      DWORD timemilliseconds = (thistime - last);
       #else
      std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds> (change).count());
       #endif
      return timemilliseconds;
    }

   double lap() {
//       #ifdef _WIN32
    #ifdef WINNOCHRONO
       DWORD thistime = getCount();
      DWORD timemilliseconds = (thistime - last);
       #else
      std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
    std::chrono::duration<double> change = thistime - last;
      double timemilliseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds> (change).count());
       #endif
      last = thistime;
      return timemilliseconds;
   }
};

