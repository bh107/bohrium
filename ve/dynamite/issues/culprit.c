#include "stdio.h"
#include "stdarg.h"
#include "stdlib.h"

int add_three(int tool, ...)
{
    va_list list;
    va_start(list, tool);

    int first_arg = va_arg(list, int);
    int second_arg = va_arg(list, int);
    int third_arg = va_arg(list, int);

    va_end(list);

    return first_arg + second_arg + third_arg;
}

/*
int main()
{
    printf("3+4+5=%d\n", add_three(0, 3,4,5));
    return 0;
}
*/

