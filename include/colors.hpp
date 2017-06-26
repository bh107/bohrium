#ifndef __COLORS
#define __COLORS

#include <unistd.h>

// Use it like:
//   std::cout << RED << "My red message" << RST << std::endl;
// Remember to issue RST to reset color.

#ifndef USE_COLORS
#define USE_COLORS isatty(1)
#endif

#define RST (USE_COLORS ? "\x1B[0m" : "")

#define RED (USE_COLORS ? "\x1B[31m" : "")
#define GRN (USE_COLORS ? "\x1B[32m" : "")
#define YEL (USE_COLORS ? "\x1B[33m" : "")
#define BLU (USE_COLORS ? "\x1B[34m" : "")
#define MAG (USE_COLORS ? "\x1B[35m" : "")
#define CYN (USE_COLORS ? "\x1B[36m" : "")
#define WHT (USE_COLORS ? "\x1B[37m" : "")

#define BOLD (USE_COLORS ? "\x1B[1m" : "")
#define UNDL (USE_COLORS ? "\x1B[4m" : "")

#endif  // __COLORS
