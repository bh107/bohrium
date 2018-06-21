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

#include <string>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/algorithm/string.hpp>
#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <bh_config_parser.hpp>

#define HOME_INI_PATH "~/.bohrium/config.ini"
#define SYSTEM_INI_PATH_1 "/usr/local/etc/bohrium/config.ini"
#define SYSTEM_INI_PATH_2 "/usr/etc/bohrium/config.ini"

using namespace std;
using namespace boost;

namespace bohrium {

namespace {
// Path to the config file e.g. ~/.bohrium/config.ini
string get_config_path() {

    const char *homepath = HOME_INI_PATH;
    const char *syspath1 = SYSTEM_INI_PATH_1;
    const char *syspath2 = SYSTEM_INI_PATH_2;

    //
    // Find the configuration file
    //

    // Start by looking a path set via environment variable.
    const char *env = getenv("BH_CONFIG");
    if (env != NULL) {
        FILE *fp = fopen(env, "r");
        if (fp)
            fclose(fp);
        else
            env = NULL;//Did not exist.
    }

    // Then the home directory.
    if (env == NULL) {
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
        char *h = getenv("HOME");
        if (h != NULL) {
            char _expand_buffer[PATH_MAX];
            snprintf(_expand_buffer, PATH_MAX, "%s/%s", h, homepath + 1);
            homepath = _expand_buffer;
        }
#endif
        FILE *fp = fopen(homepath, "r");
        if (fp) {
            env = homepath;
            fclose(fp);
        }
    }

    //And then system-wide.
    if (env == NULL) {
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
        FILE *fp = fopen(syspath1, "r");
        if (fp) {
            env = syspath1;
            fclose(fp);
        }
    }

    //And then system-wide.
    if (env == NULL) {
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
        FILE *fp = fopen(syspath2, "r");
        if (fp) {
            env = syspath2;
            fclose(fp);
        }
    }
    // We could not find the configuration file anywhere
    if (env == NULL) {
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

// Return section/option as an environment variable
// or the empty string if the environment variable wasn't found
string lookup_env(const string &section, const string &option) {
    string s = "BH_" + section + "_" + option;
    to_upper(s);
    std::replace(s.begin(), s.end(), '-', '_'); // replace all '-' to '_'
    std::replace(s.begin(), s.end(), ' ', '_'); // replace all ' ' to '_'
    const char *env = getenv(s.c_str());

    if (env == NULL) {
        return string();
    } else {
        return string(env);
    }
}

}// namespace unnamed

string ConfigParser::lookup(const string &section, const string &option) const {
    //Check environment variable
    string ret = lookup_env(section, option);
    if (not ret.empty())
        return ret;

    //Check config file
    ret = _config.get<string>(section + "." + option);

    //Remove quotes "" or '' and return
    if (ret.find_first_of("\"'") == 0 and ret.find_last_of("\"'") == ret.size() - 1) {
        return ret.substr(1, ret.size() - 2);
    } else {
        return ret;
    }
}

ConfigParser::ConfigParser(int stack_level) : file_path(get_config_path()),
                                              file_dir(boost::filesystem::path(file_path).remove_filename()),
                                              stack_level(stack_level) {

    // Load the bohrium configuration file
    property_tree::ini_parser::read_ini(file_path.string(), _config);

    // Find the stack name specified by 'BH_STACK'
    const char *env = getenv("BH_STACK");
    string stack_name;
    if (env == nullptr) {
        stack_name = "default";
    } else {
        stack_name = env;
    }
    // Read stack, which is a comma separated list of component names,
    // into a vector of component names.
    _stack_list = getList("stacks", stack_name);
    if (stack_level >= static_cast<int>(_stack_list.size()) or stack_level < -1) {
        throw ConfigError("ConfigParser: stack level is out of bound");
    }
    if (stack_level == -1) {
        _default_section = "bridge";
    } else {
        _default_section = _stack_list[stack_level];
    }
}

boost::filesystem::path ConfigParser::expand(boost::filesystem::path path) const {
    if (path.empty())
        return path;

    string s = path.string();
    if (s[0] == '~') {
        const char *home = getenv("HOME");
        if (home == nullptr) {
            throw std::invalid_argument("Couldn't expand `~` since $HOME environment variable not set.");
        }
        return boost::filesystem::path(home) / boost::filesystem::path(s.substr(1));
    } else {
        return path;
    }
}

vector<string> ConfigParser::getList(const std::string &section,
                                     const std::string &option) const {
    vector<string> ret;
    string s = get<string>(section, option);
    algorithm::split(ret, s, is_any_of("\t, "), token_compress_on);
    return ret;
}

vector<boost::filesystem::path> ConfigParser::getListOfPaths(const std::string &section,
                                                             const std::string &option) const {
    vector<boost::filesystem::path> ret;
    for (const string &path_str: getList(section, option)) {
        const auto path = expand(boost::filesystem::path(path_str));
        if (path.is_absolute() or path.empty()) {
            ret.push_back(path);
        } else {
            ret.push_back(file_dir / path);
        }
    }
    return ret;
}

string ConfigParser::getChildLibraryPath() const {
    // Do we have a child?
    if (static_cast<int>(_stack_list.size()) <= stack_level + 1) {
        return string();
    }
    // Our child is our stack level plus one
    const string child_name = _stack_list[stack_level + 1];
    return get<boost::filesystem::path>(child_name, "impl").string();
}

} //namespace bohrium
