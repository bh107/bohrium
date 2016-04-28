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
#include <string>

namespace bohrium {

//Representation of the Bohrium configuration file
class ConfigParser {
  private:
    //Path to the config file e.g. ~/.bohrium/config.ini
    std::string _file_path;
    //Default section, which will be used when no section
    //are given to the get() method.
    std::string _default_section;
    //The config data
    boost::property_tree::ptree _config;
  public:
    ConfigParser(const std::string &default_section);

    /* Get an value of the 'option' within the 'section'
     *
     * @section  The ini section e.g. [gpu]. If omitted, the default
     *           section is used.
     * @option   The ini option e.g. timing = True
     * @return   The value, which is lexically converted to type 'T'
     */
    template<typename T>
    const T& get(const std::string &section, const std::string &option) const;
    template<typename T>
    const T& get(const std::string &option) const {
        return get<T>(_default_section, option);
    }





};

} //namespace bohrium

#endif

