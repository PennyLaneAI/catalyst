cmake_minimum_required(VERSION 3.20)
project(catalyst)

set(LOGO [=[
░█▀▀░█▀█░▀█▀░█▀█░█░░░█░█░█▀▀░▀█▀
░█░░░█▀█░░█░░█▀█░█░░░░█░░▀▀█░░█░
░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░░▀░░▀▀▀░░▀░
]=])
message(${LOGO})

add_subdirectory(frontend)
add_subdirectory(mlir)
add_subdirectory(runtime)
