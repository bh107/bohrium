#ifndef __COLORS
#define __COLORS

// Use it like:
//   std::cout << RED << "My red message" << RST << std::endl;
// Remember to issue RST to reset color.

#define RST "\x1B[0m"

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"

#define BOLD "\x1B[1m"
#define UNDL "\x1B[4m"

#endif  // __COLORS
