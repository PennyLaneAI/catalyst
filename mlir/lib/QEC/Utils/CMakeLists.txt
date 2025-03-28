include(FetchContent)

FetchContent_Declare(stim
    GIT_REPOSITORY https://github.com/quantumlib/stim.git
    GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(stim)

add_library(QECUtils
    PauliStringWrapper.cpp
)

# features required by Stim
target_compile_features(QECUtils PRIVATE cxx_std_20)
target_compile_options(QECUtils PRIVATE
    -frtti           # Enable RTTI
    -fexceptions     # Enable exceptions
)

target_link_libraries(QECUtils PRIVATE libstim)
