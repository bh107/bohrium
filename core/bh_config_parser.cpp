/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/algorithm/string.hpp>
#include <cstdlib>

#include <bh_config_parser.hpp>

#ifdef _WIN32

#include <windows.h>
#include <dlfcn-win32.h>

#define HOME_INI_PATH "%APPDATA%\\bohrium\\config.ini"
#define SYSTEM_INI_PATH_1 "%PROGRAMFILES%\\bohrium\\config.ini"
#define SYSTEM_INI_PATH_2 "%PROGRAMFILES%\\bohrium\\config.ini"

//Nasty function renaming
#define snprintf _snprintf
#define strcasecmp _stricmp

#else

#include <dlfcn.h>
#include <limits.h>

#define HOME_INI_PATH "~/.bohrium/config.ini"
#define SYSTEM_INI_PATH_1 "/usr/local/etc/bohrium/config.ini"
#define SYSTEM_INI_PATH_2 "/usr/etc/bohrium/config.ini"

#endif

using namespace std;
using namespace boost;

namespace bohrium {

namespace {
// Path to the config file e.g. ~/.bohrium/config.ini
string get_config_path() {

    const char* homepath = HOME_INI_PATH;
    const char* syspath1 = SYSTEM_INI_PATH_1;
    const char* syspath2 = SYSTEM_INI_PATH_2;

    //
    // Find the configuration file
    //

    // Start by looking a path set via environment variable.
    const char *env = getenv("BH_CONFIG");
    if (env != NULL)
    {
        FILE *fp = fopen(env,"r");
        if( fp )
            fclose(fp);
        else
            env = NULL;//Did not exist.
    }

    // Then the home directory.
    if(env == NULL)
    {
#if _WIN32
        char _expand_buffer[MAX_PATH];
        DWORD result = ExpandEnvironmentStrings(
            homepath,
            _expand_buffer,
            MAX_PATH-1
        );

        if (result != 0)
        {
            homepath = _expand_buffer;
        }
#else
        char* h = getenv("HOME");
        if (h != NULL)
        {
            char _expand_buffer[PATH_MAX];
            snprintf(_expand_buffer, PATH_MAX, "%s/%s", h, homepath+1);
            homepath = _expand_buffer;
        }
#endif
        FILE *fp = fopen(homepath,"r");
        if( fp ) {
            env = homepath;
            fclose(fp);
        }
    }

    //And then system-wide.
    if(env == NULL)
    {
#if _WIN32
        char _expand_buffer[MAX_PATH];
        DWORD result = ExpandEnvironmentStrings(
            syspath1,
            _expand_buffer,
            MAX_PATH-1
        );

        if(result != 0)
        {
            syspath1 = _expand_buffer;
        }
#endif
        FILE *fp = fopen(syspath1,"r");
        if(fp)
        {
            env = syspath1;
            fclose(fp);
        }
    }

    //And then system-wide.
    if(env == NULL)
    {
#if _WIN32
        char _expand_buffer[MAX_PATH];
        DWORD result = ExpandEnvironmentStrings(
            syspath2,
            _expand_buffer,
            MAX_PATH-1
        );

        if(result != 0)
        {
            syspath2 = _expand_buffer;
        }
#endif
        FILE *fp = fopen(syspath2,"r");
        if(fp)
        {
            env = syspath2;
            fclose(fp);
        }
    }
    // We could not find the configuration file anywhere
    if(env == NULL)
    {
        fprintf(stderr, "Error: Bohrium could not find the config file.\n"
            " The search is:\n"
            "\t* The environment variable BH_CONFIG.\n"
            "\t* The home directory \"%s\".\n"
            "\t* The local directory \"%s\".\n"
            "\t* And system-wide \"%s\".\n", homepath, syspath1, syspath2);
        throw invalid_argument("No config file");
    }
    return string(env);
}
}// namespace unnamed

string ConfigParser::lookup_env(const string &section, const string &option) const
{
    string s = "BH_" + section + option;
    to_lower(s);
    const char *env = getenv(s.c_str());
    if (env == NULL) {
        return string();
    } else {
        return string(env);
    }
}

ConfigParser::ConfigParser(unsigned int stack_level) : file_path(get_config_path()),
                                                       stack_level(stack_level) {

    // Load the bohrium configuration file
    property_tree::ini_parser::read_ini(file_path, _config);

    // Find the stack name specified by 'BH_STACK'
    const char *env = getenv("BH_STACK");
    string stack_name;
    if (env == NULL){
        stack_name = "default";
    } else {
        stack_name = env;
    }
    // Read stack, which is a comma separated list of component names,
    // into a vector of component names.
    {
        string s = get<string>("stacks", stack_name);
        algorithm::split(_stack_list, s, is_any_of("\t, "), token_compress_on);
    }
    if (_stack_list.size() < stack_level) {
        throw invalid_argument("ConfigParser: stack level is out of bound");
    }
}

string ConfigParser::getChildLibraryPath() const
{
    // Do we have a child?
    if (_stack_list.size() <= stack_level+1) {
        throw runtime_error("ConfigParser: No child");
    }
    // Our child is our stack level plus one
    string child_name = _stack_list[stack_level+1];
    return get<string>(child_name, "impl");
}

} //namespace bohrium
