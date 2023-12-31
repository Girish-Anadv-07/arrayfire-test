# - Find the LAPACKE library
#
# Usage:
#   FIND_PACKAGE(LAPACKE [REQUIRED] [QUIET] )
#

INCLUDE(FindPackageHandleStandardArgs)
SET(LAPACKE_ROOT_DIR CACHE STRING
  "Root directory for custom LAPACK implementation")

if(NOT LAPACKE_ROOT_DIR)
  if (ENV{LAPACKEDIR})
    SET(LAPACKE_ROOT_DIR $ENV{LAPACKEDIR})
  endif()

  if (ENV{LAPACKE_ROOT_DIR})
    SET(LAPACKE_ROOT_DIR $ENV{LAPACKE_ROOT_DIR})
  endif()

  if (ENV{MKLROOT})
    SET(LAPACKE_ROOT_DIR $ENV{MKLROOT})
  endif()
endif()

# Check if we can use PkgConfig
FIND_PACKAGE(PkgConfig)

#Determine from PKG
IF(PKG_CONFIG_FOUND AND NOT LAPACKE_ROOT_DIR)
  PKG_CHECK_MODULES( PC_LAPACKE QUIET "lapacke")
ENDIF()

IF(PC_LAPACKE_FOUND)
    FOREACH(PC_LIB ${PC_LAPACKE_LIBRARIES})
      FIND_LIBRARY(${PC_LIB}_LIBRARY NAMES ${PC_LIB} HINTS ${PC_LAPACKE_LIBRARY_DIRS} )
      IF (NOT ${PC_LIB}_LIBRARY)
        MESSAGE(FATAL_ERROR "Something is wrong in your pkg-config file - lib ${PC_LIB} not found in ${PC_LAPACKE_LIBRARY_DIRS}")
      ENDIF (NOT ${PC_LIB}_LIBRARY)
      LIST(APPEND LAPACKE_LIB ${${PC_LIB}_LIBRARY})
    ENDFOREACH(PC_LIB)

    FIND_PATH(
        LAPACKE_INCLUDES
        NAMES "lapacke.h"
        PATHS
        ${PC_LAPACKE_INCLUDE_DIRS}
        ${INCLUDE_INSTALL_DIR}
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include
        DOC "LAPACKE Include Directory"
        )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LAPACKE DEFAULT_MSG LAPACKE_LIB)
    MARK_AS_ADVANCED(LAPACKE_INCLUDES LAPACKE_LIB)

ELSE(PC_LAPACKE_FOUND)

    IF ("${SIZE_OF_VOIDP}" EQUAL 8)
        SET(MKL_LIB_DIR_SUFFIX "intel64")
    ELSE()
        SET(MKL_LIB_DIR_SUFFIX "ia32")
    ENDIF()

    IF(LAPACKE_ROOT_DIR)
        #find libs
        FIND_LIBRARY(
            LAPACKE_LIB
            NAMES "lapacke" "LAPACKE" "liblapacke" "mkl_rt"
            PATHS ${LAPACKE_ROOT_DIR}
            PATH_SUFFIXES "lib" "lib64" "lib/${MKL_LIB_DIR_SUFFIX}"
            DOC "LAPACKE Library"
            NO_DEFAULT_PATH
            )
        FIND_PATH(
            LAPACKE_INCLUDES
            NAMES "lapacke.h" "mkl_lapacke.h"
            PATHS ${LAPACKE_ROOT_DIR}
            PATH_SUFFIXES "include"
            DOC "LAPACKE Include Directory"
            NO_DEFAULT_PATH
            )
    ELSE()
        FIND_LIBRARY(
            LAPACKE_LIB
            NAMES "lapacke" "liblapacke" "openblas" "mkl_rt"
            PATHS
            ${PC_LAPACKE_LIBRARY_DIRS}
            ${LIB_INSTALL_DIR}
            /opt/intel/mkl/lib/${MKL_LIB_DIR_SUFFIX}
            /usr/lib64
            /usr/lib
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            DOC "LAPACKE Library"
            )
        FIND_PATH(
            LAPACKE_INCLUDES
            NAMES "lapacke.h" "mkl_lapacke.h"
            PATHS
            ${PC_LAPACKE_INCLUDE_DIRS}
            ${INCLUDE_INSTALL_DIR}
            /opt/intel/mkl/include
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            DOC "LAPACKE Include Directory"
            PATH_SUFFIXES
            lapacke
            )
    ENDIF(LAPACKE_ROOT_DIR)
    find_package_handle_standard_args(LAPACKE DEFAULT_MSG LAPACKE_LIB LAPACKE_INCLUDES)
ENDIF(PC_LAPACKE_FOUND)

MARK_AS_ADVANCED(
  LAPACKE_ROOT_DIR
  LAPACKE_INCLUDES
  LAPACKE_LIB
  lapacke_LIBRARY)

if(PC_LAPACKE_FOUND OR (LAPACKE_LIB AND LAPACKE_INCLUDES))
  add_library(LAPACKE::LAPACKE UNKNOWN IMPORTED)
  set_target_properties(LAPACKE::LAPACKE PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      IMPORTED_LOCATION "${LAPACKE_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDES}"
    )
endif()
