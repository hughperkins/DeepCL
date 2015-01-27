#pragma once

#if defined(_WIN32) 
# if defined(ClConvolve_EXPORTS)
#  define ClConvolve_EXPORT __declspec(dllexport)
# else
#  define ClConvolve_EXPORT __declspec(dllimport)
# endif // ClConvolve_EXPORTS
#else // _WIN32
# define ClConvolve_EXPORT
#endif

