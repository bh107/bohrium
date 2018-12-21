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

#include <sstream>
#include <stdexcept>
#include <boost/algorithm/string/replace.hpp>
#include <jitk/compiler.hpp>
#include <jitk/subprocess.hpp>

using namespace std;
namespace P = subprocess;

namespace bohrium {
namespace jitk {


/** Returns the command where {OUT} and {IN} are expanded. */
string expand_compile_cmd(const string &cmd_template, const string &out, const string &in, const string &config_path) {
    string ret = cmd_template;
    boost::replace_all(ret, "{OUT}", out);
    boost::replace_all(ret, "{IN}", in);
    return ret;
}


void Compiler::compile(const boost::filesystem::path &output_file, const string &source,
                       const string &command) const {
    const string cmd = expand_compile_cmd(command, output_file.string(), " - ", config_path);
    if (verbose) {
        cout << "compile command: \"" << cmd << "\"" << endl;
    }
    P::Popen p = P::Popen(cmd, P::input{P::PIPE}, P::output{P::PIPE}, P::error{P::PIPE});
    p.send(source.c_str(), source.size());
    auto res = p.communicate();
    stringstream ss;
    ss << "[JIT compiler fatal error retcode: " << p.retcode() << "]\n";
    ss << res.first.buf.data() << "\n"; // Stdout
    ss << res.second.buf.data() << "\n";// Stderr
    if (p.retcode() > 0) {
        throw std::runtime_error(ss.str());
    }
}

void Compiler::compile(const boost::filesystem::path &output_file, const boost::filesystem::path &source_file, const std::string &command) const {
    const string cmd = expand_compile_cmd(command, output_file.string(), source_file.string(), config_path);
    if (verbose) {
        cout << "compile command: \"" << cmd << "\"" << endl;
    }
    P::Popen p = P::Popen(cmd, P::output{P::PIPE}, P::error{P::PIPE});
    auto res = p.communicate();
    stringstream ss;
    ss << "[JIT compiler fatal error retcode: " << p.retcode() << "]\n";
    ss << res.first.buf.data() << "\n"; // Stdout
    ss << res.second.buf.data() << "\n";// Stderr
    if (p.retcode() > 0) {
        throw std::runtime_error(ss.str());
    }
}

}
}
