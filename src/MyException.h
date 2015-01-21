// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdio>
#include <string>
using namespace std;

class MyException : public exception {
   char msg[255];
public:
   MyException( string msg ) {
      sprintf( this->msg, "%.254s", msg.c_str() );
   }
   virtual const char *what() const throw() {
       return msg;
   }
};


