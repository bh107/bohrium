cmake_minimum_required(VERSION 2.6)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_fftw SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_fftw bh)

#External dependencies
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIR})
set(LIBS ${LIBS} ${FFTW_LIBRARIES})
target_link_libraries(bh_fftw ${LIBS})

install(TARGETS bh_fftw DESTINATION lib)
