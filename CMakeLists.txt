cmake_minimum_required(VERSION 3.18)
project(torchfx LANGUAGES CXX)

# PyTorch's public C++ headers require C++17. Linux GCC 11+ defaults to it,
# but Apple clang on the macOS runners does not, so the build of
# `torch/extension.h` fails with cryptic template errors. Force the standard
# explicitly here for every target the project defines below.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Find Python & PyTorch ------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Query torch for include/library paths (works regardless of torch's CUDA
# build status — unlike find_package(Torch) which requires a CUDA toolkit
# when torch was compiled with CUDA).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import torch.utils.cpp_extension as e; print(';'.join(e.include_paths()))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import torch.utils.cpp_extension as e; print(';'.join(e.library_paths()))"
  OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch library dirs: ${TORCH_LIBRARY_DIRS}")
message(STATUS "Torch CXX11 ABI: ${TORCH_CXX11_ABI}")

# ---------- Optional CUDA --------------------------------------------------
option(TORCHFX_USE_CUDA "Build CUDA kernels" ON)

if(DEFINED ENV{TORCHFX_NO_CUDA} AND NOT "$ENV{TORCHFX_NO_CUDA}" STREQUAL "")
  set(TORCHFX_USE_CUDA OFF)
endif()

if(TORCHFX_USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      message(STATUS "CUDA toolkit not found -- building CPU-only extension")
      set(TORCHFX_USE_CUDA OFF)
    endif()
  else()
    message(STATUS "No CUDA compiler found -- building CPU-only extension")
    set(TORCHFX_USE_CUDA OFF)
  endif()
endif()

message(STATUS "TORCHFX_USE_CUDA = ${TORCHFX_USE_CUDA}")

# ---------- Extension target -----------------------------------------------
set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/torchfx/_csrc")

Python_add_library(torchfx_ext MODULE WITH_SOABI
  "${CSRC_DIR}/binding.cpp"
  "${CSRC_DIR}/cpu/iir_cpu.cpp"
  "${CSRC_DIR}/cpu/delay_cpu.cpp")

if(TORCHFX_USE_CUDA)
  target_sources(torchfx_ext PRIVATE
    "${CSRC_DIR}/cuda/parallel_scan.cu"
    "${CSRC_DIR}/cuda/biquad_forward.cu"
    "${CSRC_DIR}/cuda/delay_forward.cu")
  target_compile_definitions(torchfx_ext PRIVATE WITH_CUDA)
endif()

target_include_directories(torchfx_ext PRIVATE
  "${CSRC_DIR}/include"
  ${TORCH_INCLUDE_DIRS})

target_compile_definitions(torchfx_ext PRIVATE
  TORCH_EXTENSION_NAME=torchfx_ext
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

target_link_directories(torchfx_ext PRIVATE ${TORCH_LIBRARY_DIRS})
target_link_libraries(torchfx_ext PRIVATE torch torch_python torch_cpu c10)

target_compile_features(torchfx_ext PRIVATE cxx_std_17)

# Compiler-specific optimisation and standard flags. We avoid `-march=native`
# / `/arch:` so wheels stay portable across CPUs.
#
# - GCC / Clang / AppleClang: `-O3 -ffast-math`.
# - MSVC: `/O2 /fp:fast`, plus an explicit `/std:c++17` because some
#   torch headers use C++17-only constructs (nested namespaces,
#   `std::optional`, `std::string_view`) that MSVC otherwise rejects with
#   "language feature requires compiler flag '/std:c++17'", and
#   `/Zc:__cplusplus` so torch's feature-detection macros see the right
#   `__cplusplus` value (MSVC defaults to reporting 199711L unless this
#   conformance flag is set).
target_compile_options(torchfx_ext PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -ffast-math>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/O2 /fp:fast /std:c++17 /Zc:__cplusplus>
)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(torchfx_ext PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install into the torchfx package so `from torchfx import torchfx_ext` works.
install(TARGETS torchfx_ext LIBRARY DESTINATION torchfx)
