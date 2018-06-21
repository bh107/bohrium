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
#pragma once

#include <boost/property_tree/ptree.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>

// We need to specialize lexical_cast() in order to support booleans
// other then the standard 0/1 to true/false conversion.
namespace boost {
template<>
inline bool lexical_cast<bool, std::string>(const std::string &arg) {
    switch (arg[0]) {
        case 'y':
        case 'Y':
        case '1':
        case 't':
        case 'T':
            return true;
        case 'n':
        case 'N':
        case '0':
        case 'f':
        case 'F':
            return false;
        default:
            throw boost::bad_lexical_cast();
    }
}
}

namespace bohrium {

// Generic Config Parser Exception
class ConfigError : public std::exception {
    std::string _msg;
public:
    ConfigError(const std::string &msg) : _msg(msg) {}

    virtual const char *what() const throw() { return _msg.c_str(); }
};

// Exception thrown when getChildLibraryPath() does not exist
class ConfigNoChild : public ConfigError {
public:
    ConfigNoChild(const std::string &msg) : ConfigError(msg) {}
};

// Exception thrown when a config key is not found
class ConfigKeyNotFound : public ConfigError {
    std::string _msg;
public:
    ConfigKeyNotFound(const std::string &msg) : ConfigError(msg) {}
};

// Exception thrown when cast failed
class ConfigBadCast : public ConfigError {
    std::string _msg;
public:
    ConfigBadCast(const std::string &msg) : ConfigError(msg) {}
};

//Representation of the Bohrium configuration file
class ConfigParser {
public:
    // Path to the config file e.g. ~/.bohrium/config.ini
    const boost::filesystem::path file_path;
    // Path to the directory of the config file
    const boost::filesystem::path file_dir;
    // The stack level of the calling component (-1 is the bridge,
    // 0 is the first component in the stack list, 1 is the second component etc.)
    const int stack_level;
private:
    // The default section, which should be name of the component owning this class
    std::string _default_section;
    // The list of components in the user-specified stack starting
    // at the bridge
    std::vector<std::string> _stack_list;
    // The config data
    boost::property_tree::ptree _config;

    // Return section/option first looking at the environment variable
    // and then the ini file.
    std::string lookup(const std::string &section,
                       const std::string &option) const;

public:
    /** Uses 'stack_level' to find the default section to use with get()
     * and when calculating the child in getChild()
     *
     * @stack_level  Is the level of the calling component
     */
    ConfigParser(int stack_level);

    /** Expand `~` in `path` to the home dir of the user
     *
     * @path    Path to expand.
     * @return  The expanded path.
     */
    boost::filesystem::path expand(boost::filesystem::path path) const;

    /** Get the value of the 'option' within the 'section'
     *
     * @section  The ini section e.g. [gpu]. If omitted, the default
     *           section is used.
     * @option   The ini option e.g. timing = True
     * @return   The value, which is lexically converted to type 'T'
     * Throws ConfigKeyNotFound if the section/option does not exist
     * Throws ConfigBadCast if the value cannot be converted
     */
    template<typename T>
    T get(const std::string &section, const std::string &option) const {
        using namespace std;
        using namespace boost;
        //Retrieve the option
        string ret;
        try {
            ret = lookup(section, option);
        } catch (const property_tree::ptree_bad_path &) {
            stringstream ss;
            ss << "Error parsing the config file '" << file_path << "': '" << section << "." << option
               << "' not found!\n Using an old config file? try removing it and re-install bohrium" << endl;
            throw ConfigKeyNotFound(ss.str());
        }
        //Now let's try to convert the value to the requested type
        try {
            return lexical_cast<T>(ret);
        } catch (const boost::bad_lexical_cast &) {
            stringstream ss;
            ss << "ConfigParser cannot convert '" << section << "." << option
               << "=" << ret << "' to type <" << typeid(T).name() << ">" << endl;
            throw ConfigBadCast(ss.str());
        }
    }

    // Overload that use the default section
    template<typename T>
    T get(const std::string &option) const {
        return get<T>(_default_section, option);
    }

    /** Get the value of the 'option' within the 'section' and if it
     * does not exist return 'default_value' instead
     *
     * @section        The ini section e.g. [gpu]. If omitted, the
     *                 default section is used.
     * @option         The ini option e.g. timing = True
     * @default_value  The default value of type 'T'
     * @return         The value, which is lexically converted to
     *                 type 'T' or the 'default_value'
     */
    template<typename T>
    T defaultGet(const std::string &section, const std::string &option,
                 const T &default_value) const {
        try {
            return get<T>(section, option);
        } catch (const ConfigKeyNotFound &) {
            return default_value;
        }
    }

    template<typename T>
    T defaultGet(const std::string &option, const T &default_value) const {
        return defaultGet(_default_section, option, default_value);
    }

    /** Get the value of the 'option' within the 'section' and convert the value,
     * which must be a comma separated list, into a vector of strings.
     *
     * @section        The ini section e.g. [gpu]. If omitted, the
     *                 default section is used.
     * @option         The ini option e.g. timing = True
     * @return         Vector of strings
     * Throws ConfigKeyNotFound if the section/option does not exist
     */
    std::vector<std::string> getList(const std::string &section,
                                     const std::string &option) const;

    std::vector<std::string> getList(const std::string &option) const {
        return getList(_default_section, option);
    }

    /** Get the value of the 'option' within the 'section' and convert the value,
     * which must be a comma separated list, into a vector of paths.
     *
     * @section        The ini section e.g. [gpu]. If omitted, the
     *                 default section is used.
     * @option         The ini option e.g. timing = True
     * @return         Vector of paths
     * Throws ConfigKeyNotFound if the section/option does not exist
     */
    std::vector<boost::filesystem::path> getListOfPaths(const std::string &section,
                                                        const std::string &option) const;

    std::vector<boost::filesystem::path> getListOfPaths(const std::string &option) const {
        return getListOfPaths(_default_section, option);
    }

    /** Get the value of the 'option' within the 'section' and convert the value,
     * which must be a comma separated list, into a vector of strings.
     * If it does not exist return 'default_value' instead.
     *
     * @section        The ini section e.g. [gpu]. If omitted, the
     *                 default section is used.
     * @option         The ini option e.g. timing = True
     * @default_value  The default value
     * @return         Vector of strings
     * Throws ConfigKeyNotFound if the section/option does not exist
     */
    std::vector<std::string> defaultGetList(const std::string &section, const std::string &option,
                                            const std::vector<std::string> &default_value) const {
        try {
            return getList(section, option);
        } catch (const ConfigKeyNotFound &) {
            return default_value;
        }
    }

    std::vector<std::string> defaultGetList(const std::string &option,
                                            const std::vector<std::string> &default_value) const {
        return defaultGetList(_default_section, option, default_value);
    }

    /** Return the path to the library that implements the calling component's child or "" if the child doesn't exist.
     *
     * @return File path to shared library or the empty string
     * Throw ConfigNoChild exception if the calling component has not children
     */
    std::string getChildLibraryPath() const;

    /** Retrieve the name of the calling component
     *
     * @return Component name as given in the config file
     */
    std::string getName() const { return _default_section; };
};

// Path specialization of `ConfigParser::get()`, which makes sure that relative paths are converted to absolute paths
template<>
inline boost::filesystem::path ConfigParser::get(const std::string &section, const std::string &option) const {
    boost::filesystem::path ret = expand(boost::filesystem::path(get<std::string>(section, option)));
    if (ret.is_absolute() or ret.empty()) {
        return ret;
    } else {
        return file_dir / ret;
    }
}

} //namespace bohrium
