#pragma once

#include "DeepCLDllExport.h"

#include <string>

// handles helping to call across cpp runtime boundaries

// allocates new string, returns it.  MUST call deleteCharStar to delete it
DeepCL_EXPORT const char *deepcl_stringToCharStar(std::string astring);
DeepCL_EXPORT void deepcl_deleteCharStar(const char *charStar);

