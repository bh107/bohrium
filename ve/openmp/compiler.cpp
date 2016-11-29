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

#include "compiler.hpp"

using namespace std;

namespace bohrium {

Compiler::Compiler(string cmd, string inc, string lib, string flg, string ext) : cmd_(cmd), inc_(inc), lib_(lib), flg_(flg), ext_(ext) {}

string Compiler::text() const {
    stringstream ss;
    ss << "Compiler {" << endl;
    ss << "  cmd = '" << cmd_ << "'," << endl;
    ss << "  inc = '" << inc_ << "'," << endl;
    ss << "  lib = '" << lib_ << "'," << endl;
    ss << "  flg = '" << flg_ << "'," << endl;
    ss << "  ext = '" << ext_ << "'," << endl;
    ss << "  process_str = '" << process_str("OBJ", "SRC") << "'" << endl;
    ss << "}";
    return ss.str();
}

string Compiler::process_str(string object_abspath, string source_abspath) const {
    stringstream ss;

    ss           << cmd_;
    ss << " "    << inc_; 
    ss << " "    << flg_;
    ss << " "    << ext_;
    ss << " "    << source_abspath;
    ss << " "    << lib_;
    ss << " -o " << object_abspath;

    return ss.str();
}

void Compiler::compile(string object_abspath, const char* sourcecode, size_t source_len) const {
    string cmd = process_str(object_abspath, " - ");
//    cout << "compile command: " << cmd << endl;

    FILE* cmd_stdin = popen(cmd.c_str(), "w");  // Open process and get stdin stream
    if (!cmd_stdin) {
        perror("popen()");
        fprintf(stderr, "popen() failed for: [%s]", sourcecode);
        pclose(cmd_stdin);
        throw runtime_error("Compiler: popen() failed");
    }
                                                // Write / pipe to stdin
    int write_res = fwrite(sourcecode, sizeof(char), source_len, cmd_stdin);
    if (write_res < (int)source_len) {
        perror("fwrite()");
        fprintf(stderr, "fwrite() failed in file %s at line # %d\n", __FILE__, __LINE__-5);
        pclose(cmd_stdin);
        throw runtime_error("Compiler: error!");
    }

    int flush_res = fflush(cmd_stdin);          // Flush stdin
    if (EOF == flush_res) {
        perror("fflush()");
        fprintf(stderr, "fflush() failed in file %s at line # %d\n", __FILE__, __LINE__-5);
        pclose(cmd_stdin);
        throw runtime_error("Compiler: fflush() failed");
    }

    int exit_code = (pclose(cmd_stdin)/256);
    if (0!=exit_code) {
        perror("pclose()");
        fprintf(stderr, "pclose() failed.\n");
        throw runtime_error("Compiler: pclose() failed");
    }
}

void Compiler::compile(string object_abspath, string src_abspath) const {
    string cmd = process_str(object_abspath, src_abspath);
    // cout << "compile command: " << cmd << endl;

    // Execute the process
    FILE *cmd_stdin = NULL;                     // Handle for library-file
    cmd_stdin = popen(cmd.c_str(), "w");        // Execute the command
    if (!cmd_stdin) {
        std::cout << "Err: Could not execute process! ["<< cmd <<"]" << std::endl;
        throw runtime_error("Compiler: error!");
    }
    fflush(cmd_stdin);
    int exit_code = (pclose(cmd_stdin)/256);
    if (0!=exit_code) {
        perror("pclose()");
        fprintf(stderr, "pclose() failed.\n");
        throw runtime_error("Compiler: pclose() failed");
    }
}

}
