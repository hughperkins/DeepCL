#pragma once

#include <string>

// handles helping to call across cpp runtime boundaries

// allocates new string, returns it.  MUST call deleteCharStar to delete it
const char *deepcl_stringToCharStar(std::string astring);
void deepcl_deleteCharStar(const char *charStar);

