// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>

#include "DeepCLDllExport.h"

DeepCL_EXPORT void arrayCopy(float *dest, float const*src, int N);
DeepCL_EXPORT void arrayZero(float *array, int N);
DeepCL_EXPORT std::string toString(float const*array, int N);


