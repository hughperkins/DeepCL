#pragma once

#include <random>

//#if (_MSC_VER == 1500 || _MSC_VER == 1600  )
#ifdef _MSC_VER // make consistent across all msvc versions, so dont have to retest on different msvc versions...
#define TR1RANDOM
typedef std::tr1::mt19937 MT19937;
#else
typedef std::mt19937 MT19937;
#endif

