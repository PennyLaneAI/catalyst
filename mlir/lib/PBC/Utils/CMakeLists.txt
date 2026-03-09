include(FetchContent)

FetchContent_Declare(stim
    GIT_REPOSITORY https://github.com/quantumlib/stim.git
    GIT_TAG v1.14.0
    SYSTEM  # ignore warnings from headers
)

# Set a more portable vectorization instruction set than whatever is available on the build machine.
set(SIMD_WIDTH 128)

FetchContent_MakeAvailable(stim)

# Ignore all warnings from Stim library sources.
target_compile_options(libstim PRIVATE -w)

add_library(PBCUtils STATIC
    PauliStringWrapper.cpp
    PBCLayer.cpp
    PBCOpUtils.cpp
)

# features required by Stim
target_compile_features(PBCUtils PRIVATE cxx_std_20)
target_compile_options(PBCUtils PRIVATE
    -frtti           # Enable RTTI
    -fexceptions     # Enable exceptions
)

target_link_libraries(PBCUtils PRIVATE libstim ExternalStablehloLib)
