/*
This file is part of Bohrium and Copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __CPP_ARGPARSE
#define __CPP_ARGPARSE

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>

namespace argparse {

typedef struct arguments {
    std::vector<size_t> size;
    bool verbose;
} arguments_t;

bool parse_args(int argc, char *argv[], arguments_t& args)
{
    int opt = 0;
    int long_index = 0;
    static struct option long_options[] = {
        {"size",    required_argument,  0, 's'},
        {"verbose", optional_argument, 0, 'v'},
        {0,0,0,0}
    };

    args.verbose = false;       // Default: verbose = false
                                
    while ((opt = getopt_long(argc, argv,"sv",
            long_options, &long_index )) != -1) {
        const char *str = optarg;
        switch (opt) {
            case 's':           // Parse --size=N_1*..*N_n
                do {
                    const char *begin = str;
                    while('*' != *str && *str) {
                        str++;
                    }
                    args.size.push_back(atoi(std::string(begin,str).c_str()));
                    
                } while(*str++ != 0);
                break;
            case 'v':           // Parse verbose
                args.verbose = true;
                break;
            default:
                return false;
        }
    }
    return true;
} 

}
#endif

