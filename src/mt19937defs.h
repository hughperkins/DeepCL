#pragma once

#include <random>

#if (_MSC_VER == 1500 || _MSC_VER == 1600  )
#define TR1RANDOM
typedef std::tr1::mt19937 MT19937;
#else
typedef std::mt19937 MT19937;
#endif


