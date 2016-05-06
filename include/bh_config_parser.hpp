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
    // Return section/option as an environment variable
    // or the empty string if the environment variable wasn't found
    std::string lookup_env(const std::string &section,
                           const std::string &option) const;
  public:
    /* Uses 'stack_level' to find the default section to use with get()
     * and when calculating the child in getChild()
     *
     * @stack_level  Is the level of the calling component
     */
    ConfigParser(unsigned int stack_level);

    /* Get an value of the 'option' within the 'section'
     *
     * @section  The ini section e.g. [gpu]. If omitted, the default
     *           section is used.
     * @option   The ini option e.g. timing = True
     * @return   The value, which is lexically converted to type 'T'
     */
    template<typename T>
    T get(const std::string &section, const std::string &option) const {
        using namespace std;
        using namespace boost;
        //Check the environment variable e.g. BH_VEM_NODE_TIMING
        {
            string env = lookup_env(section, option);
            if (not env.empty()) {
                return lexical_cast<T>(env);
            }
        }
        //Check the config file
        try {
            string s = _config.get<string>(section + "." + option);
            return lexical_cast<T>(s);
        } catch (const property_tree::ptree_bad_path&) {
            cerr << "Error parsing the config file '" << file_path << "', '"
                 << section << "." << option << "' not found!" << endl;
            throw;
        }
    }
    template<typename T>
    T get(const std::string &option) const {
        return get<T>(_stack_list[stack_level], option);
    }

    /* Get an value of the 'option' within the 'section' and if it
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
        try {
            return get<T>(option);
        } catch (const boost::property_tree::ptree_bad_path&) {
            return default_value;
        }
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

