// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <ctime>
#include <sys/time.h>

class Timer{
public:
   // clock_t last;
   double last;
   Timer() {
      //last = clock();
      last = getCount();
   }

   double getCount() {
	    struct timeval tm;
	    gettimeofday( &tm, NULL );
	    return (double)tm.tv_sec + (double)tm.tv_usec / 1000000.0;
   }

   void timeCheck(std::string label ) {
      double thistime = getCount();
      double timemilliseconds = 1000 * ( thistime - last );
      last = thistime;
      std::cout << label << " " << timemilliseconds << " ms" << std::endl;
   }

   double lap() {
      double thistime = getCount();
      double timemilliseconds = 1000 * ( thistime - last );
      last = thistime;
      return timemilliseconds;
   }
};

