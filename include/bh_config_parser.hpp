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

#ifndef __BH_CONFIG_PARSER_H
#define __BH_CONFIG_PARSER_H

#include <boost/property_tree/ptree.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

// We need to specialize lexical_cast() in order to support booleans
// other then the standard 0/1 to true/false conversion.
namespace boost {
    template<>
    inline bool lexical_cast<bool, std::string>(const std::string& arg) {
        switch(arg[0]) {
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

//Representation of the Bohrium configuration file
class ConfigParser {
  public:
    // Path to the config file e.g. ~/.bohrium/config.ini
    const std::string file_path;
    // The stack level of the calling component
    const unsigned int stack_level;
  private:
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
    /* Uses 'stack_level' to find the default section to use with get()
     * and when calculating the child in getChild()
     *
     * @stack_level  Is the level of the calling component
     */
    ConfigParser(unsigned int stack_level);

    /* Get the value of the 'option' within the 'section'
     *
     * @section  The ini section e.g. [gpu]. If omitted, the default
     *           section is used.
     * @option   The ini option e.g. timing = True
     * @return   The value, which is lexically converted to type 'T'
     * Throws property_tree::ptree_bad_path if the section/option does not exist
     * Throws bad_lexical_cast if the value cannot be converted
     */
    template<typename T>
    T get(const std::string &section, const std::string &option) const {
        using namespace std;
        using namespace boost;
        //Retrieve the option
        string ret;
        try {
            ret = lookup(section, option);
        } catch (const property_tree::ptree_bad_path&) {
            cerr << "Error parsing the config file '" << file_path << "', '"
                 << section << "." << option << "' not found!" << endl;
            throw;
        }
        //Now let's try to convert the value to the requested type
        try {
            return lexical_cast<T>(ret);
        } catch (const boost::bad_lexical_cast&) {
            string s = _config.get<string>(section + "." + option);
            cerr << "ConfigParser cannot convert '" << section << "." << option
                 << "=" << s << "' to type <" << typeid(T).name() << ">" << endl;
            throw;
        }
    }
    template<typename T>
    T get(const std::string &option) const {
        return get<T>(_stack_list[stack_level], option);
    }

    /* Get the value of the 'option' within the 'section' and if it
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
        } catch (const boost::property_tree::ptree_bad_path&) {
            return default_value;
        }
    }
    template<typename T>
    T defaultGet(const std::string &option, const T &default_value) const {
        return defaultGet(_stack_list[stack_level], option, default_value);
    }

    /* Get the value of the 'option' within the 'section' and convert the value,
     * which must be a comma separated list, into a vector of strings.
     *
     * @section        The ini section e.g. [gpu]. If omitted, the
     *                 default section is used.
     * @option         The ini option e.g. timing = True
     * @return         Vector of strings
     * Throws property_tree::ptree_bad_path if the section/option does not exist
     */
    std::vector<std::string> getList(const std::string &section,
                                     const std::string &option) const;
    std::vector<std::string> getList(const std::string &option) const {
        return getList(_stack_list[stack_level], option);
    }
    /* Return the path to the library that implements
     * the calling component's child.
     *
     * @return File path to shared library
     * Throw exception if the calling component has not children
     */
    std::string getChildLibraryPath() const;

    /* Retrieve the name of the calling component
     *
     * @return Component name as given in the config file
     */
    std::string getName() const { return _stack_list[stack_level]; };
};

} //namespace bohrium

#endif

