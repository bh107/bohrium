/*

Enable debug printing with:

gcc ... -DDEBUG

Courtesy of:

http://stackoverflow.com/questions/1941307/c-debug-print-macros
http://stackoverflow.com/questions/1644868/c-define-macro-for-debug-printing

*/
#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif
