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
#include <string>
#include <exception>

#ifndef __CPHVB_VEM_CLUSTER_EXCEPT_H
#define __CPHVB_VEM_CLUSTER_EXCEPT_H

#include <exception>
#include <sstream>

#define EXCEPT( msg ) {throw except(msg, __LINE__, __FILE__ );}
#define EXCEPT_OUT_OF_MEMORY(x) {throw except_out_of_memory(__LINE__, __FILE__ );}
#define EXCEPT_MPI(errcode) {throw except_mpi(errcode, __LINE__, __FILE__ );}


class except : public std::exception 
{
    public:
        except(const std::string & msg, int line,
               const std::string & file );

        ~except() throw();

        const char *what() const throw();

    protected:
        std::string mMsg, mFile;
        int mLine;
};

class except_out_of_memory : public except
{
    public:
        const char *what() const throw();
        except_out_of_memory(int line, const std::string & file );
};

class except_mpi : public except
{
    public:
        const char *what() const throw();
        except_mpi(int mpi_error_code, int line, const std::string & file );
    private:
        int errcode;
};


#endif
