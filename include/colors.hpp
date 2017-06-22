#ifndef __COLORS
#define __COLORS

#include <unistd.h>

// Use it like:
//   std::cout << RED << "My red message" << RST << std::endl;
// Remember to issue RST to reset color.

#define RST (isatty(1) ? "\x1B[0m" : "")

#define RED (isatty(1) ? "\x1B[31m" : "")
#define GRN (isatty(1) ? "\x1B[32m" : "")
#define YEL (isatty(1) ? "\x1B[33m" : "")
#define BLU (isatty(1) ? "\x1B[34m" : "")
#define MAG (isatty(1) ? "\x1B[35m" : "")
#define CYN (isatty(1) ? "\x1B[36m" : "")
#define WHT (isatty(1) ? "\x1B[37m" : "")

#define BOLD (isatty(1) ? "\x1B[1m" : "")
#define UNDL (isatty(1) ? "\x1B[4m" : "")

#endif  // __COLORS
