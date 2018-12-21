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

#include <string>
#include <boost/filesystem.hpp>
#include <bh_config_parser.hpp>

namespace bohrium {
namespace jitk {

/** Compiler that fork a process that compiles a shared library */
class Compiler {
public:
    std::string cmd_template;
    std::string config_path;
    bool verbose = false;

    /** Default constructor */
    Compiler() = default;

    /** Constructor that takes:
     *
     * @param cmd_template Default command that expand {OUT} and {IN}
     * @param config_path Path to the configuration file
     * @param verbose Print the fully expanded compile command
     */
    Compiler(std::string cmd_template, std::string config_path, bool verbose) : cmd_template(std::move(cmd_template)),
                                                                                config_path(std::move(config_path)),
                                                                                verbose(verbose) {}

    /** Compile source to a binary shared library
     *
     * @param output_file Path to the resulting output file
     * @param source The source to compile
     * @param command The compile command where {OUT} and {IN} are expanded.
     */
    void
    compile(const boost::filesystem::path &output_file, const std::string &source, const std::string &command) const;

    /** Compile source to a binary shared library using the default command template `cmd_template`
     *
     * @param output_file Path to the resulting output file
     * @param source The source to compile
     */
    void compile(const boost::filesystem::path &output_file, const std::string &source) const {
        compile(output_file, source, cmd_template);
    }

    /** Compile source to a binary shared library
     *
     * @param output_file Path to the resulting output file
     * @param source_file Path to the source file
     * @param command The compile command where {OUT} and {IN} are expanded.
     */
    void compile(const boost::filesystem::path &output_file, const boost::filesystem::path &source_file,
                 const std::string &command) const;

    /** Compile source to a binary shared library using the default command template `cmd_template`
     *
     * @param output_file Path to the resulting output file
     * @param source_file Path to the source file
     */
    void compile(const boost::filesystem::path &output_file, const boost::filesystem::path &source_file) const {
        compile(output_file, source_file, cmd_template);
    }
};


}
}

