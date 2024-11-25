configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DIALECT_TESTS_DEPEND
    FileCheck
    quantum-opt
    catalyst-cli
)

set(TEST_SUITES
    Quantum
    Gradient
    Mitigation
    Catalyst
    Ion
    cli
    )


if(QUANTUM_ENABLE_BINDINGS_PYTHON)
    list(APPEND DIALECT_TESTS_DEPEND QuantumPythonModules)
endif()

add_lit_testsuite(check-dialects "Run the regression tests for mlir dialects"
    ${TEST_SUITES}
    DEPENDS ${DIALECT_TESTS_DEPEND}
)

add_subdirectory(frontend)

add_custom_target(check-all
    DEPENDS
    check-dialects
    check-frontend
)
