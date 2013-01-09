/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb.h>
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
