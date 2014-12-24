#pragma once

#include <iostream>
#include <ctime>
#include <sys/time.h>
//using namespace std;

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

   void timeCheck(string label ) {
      double thistime = getCount();
      double timemilliseconds = 1000 * ( thistime - last );
      last = thistime;
      cout << label << " " << timemilliseconds << " ms" << endl;
   }

   double lap() {
      double thistime = getCount();
      double timemilliseconds = 1000 * ( thistime - last );
      last = thistime;
      return timemilliseconds;
   }
};

