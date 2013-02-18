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

#include <bh.h>
#include <sstream>
#include <mpi.h>
#include "except.h"

using std::string;

except::except(const string & msg, int line, const string & file ) 
    : mMsg( msg ), mFile( file ), mLine( line ) {}

except :: ~except() throw() {}

const char * except::what() const throw() 
{
    std::ostringstream os;    
    os << mMsg << " (" << mFile << ":" << mLine << ")" << std::endl; 

    return os.str().c_str();
}

except_out_of_memory::except_out_of_memory(int line,
                                           const std::string & file )
    : except("", line, file){}

const char * except_out_of_memory::what() const throw() 
{
    std::ostringstream os;    
    os << "Out of memory" << " (" << mFile << ":" << mLine << ")" << std::endl; 

    return os.str().c_str();
}

except_mpi::except_mpi(int mpi_error_code, int line, const std::string & file )
    : except("", line, file)
{
    errcode = mpi_error_code;
}

const char * except_mpi::what() const throw() 
{
    std::ostringstream os;
    char text[MPI_MAX_ERROR_STRING];
    int len;

    MPI_Error_string(errcode, text, &len);
 
    os << "MPI error: \"" << text << "\" (" << mFile << ":" << mLine << ")" << std::endl; 

    return os.str().c_str();
}

except_inst::except_inst(bh_opcode opcode, 
                         bh_error inst_list_status, 
                         int line, const std::string & file)
    : except("", line, file)
{
    op = opcode;
    retcode = inst_list_status;
}

const char * except_inst::what() const throw() 
{
    std::ostringstream os;
 
    os << "Error when executing " << bh_opcode_text(op);
    os << ", return code: " << bh_error_text(retcode); 
    os << " (" << mFile << ":" << mLine << ")" << std::endl; 

    return os.str().c_str();
}
