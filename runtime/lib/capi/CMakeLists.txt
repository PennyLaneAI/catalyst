##################################
# Object Lib catalyst_qir_qis_obj
##################################

add_library(catalyst_qir_qis_obj OBJECT RuntimeCAPI.cpp)

# link to rt_backend
target_link_libraries(catalyst_qir_qis_obj ${CMAKE_DL_LIBS}
                                            "-L${CMAKE_BINARY_DIR}/lib/backend"
                                            -lrt_backend
                                            lightning_dynamic)

# link to qir_stdlib
target_link_libraries(catalyst_qir_qis_obj "-L${QIR_STDLIB_LIB}"
                                            "-Wl,--whole-archive"
                                            qir_stdlib
                                            "-Wl,--no-whole-archive"
                                            pthread
                                            dl)

target_include_directories(catalyst_qir_qis_obj PUBLIC .
                                                        ${CMAKE_CURRENT_SOURCE_DIR}
                                                        ${QIR_STDLIB_INCLUDES}
                                                        ${runtime_includes}
                                                        ${backend_includes}
                                                        ${extensions_includes})

set_property(TARGET catalyst_qir_qis_obj PROPERTY POSITION_INDEPENDENT_CODE ON)


#####################
# Shared Lib rt_capi
#####################

add_library(rt_capi SHARED)

# link to rt_backend
target_link_libraries(rt_capi ${CMAKE_DL_LIBS}
                                catalyst_qir_qis_obj
                                "-L${CMAKE_BINARY_DIR}/lib/backend"
                                -lrt_backend)

include_directories(${LLVM_SRC_DIR}/mlir/include/)

# link to qir_stdlib
target_link_libraries(rt_capi "-L${QIR_STDLIB_LIB}"
	                      "-L${LLVM_BUILD_DIR}/lib/"
			      "-Wl,--rpath,${LLVM_BUILD_DIR}/lib"
			        "-lmlir_c_runner_utils"
                                "-Wl,--whole-archive"
                                qir_stdlib
                                "-Wl,--no-whole-archive"
                                pthread
                                dl)

add_dependencies(rt_capi rt_backend)

target_include_directories(rt_capi PUBLIC .
                                        ${CMAKE_CURRENT_SOURCE_DIR}
                                        ${QIR_STDLIB_INCLUDES}
                                        ${runtime_includes}
                                        ${backend_includes}
                                        ${extensions_includes})

set_property(TARGET rt_capi PROPERTY POSITION_INDEPENDENT_CODE ON)
