# - Find Jupyter
#
# defines
#
# JUPYTER_FOUND - system has jupyter
# JUPYTER_EXECUTABLE - where to find 'jupyter'

set_package_properties(jupyter PROPERTIES DESCRIPTION "Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text." URL "jupyter.org/")
set_package_properties(jupyter PROPERTIES TYPE OPTIONAL PURPOSE "Copies the Jupyter Bohrium startup script")

FIND_PROGRAM (JUPYTER_EXECUTABLE jupyter)

SET (JUPYTER_FOUND FALSE CACHE INTERNAL "")

if(JUPYTER_EXECUTABLE)
    SET (JUPYTER_FOUND TRUE CACHE INTERNAL "")
endif()

MARK_AS_ADVANCED(JUPYTER_EXECUTABLE)
