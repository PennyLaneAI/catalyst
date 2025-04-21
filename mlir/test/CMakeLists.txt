configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
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
    MBQC
    cli
    QEC
    )


if(QUANTUM_ENABLE_BINDINGS_PYTHON)
    list(APPEND DIALECT_TESTS_DEPEND QuantumPythonModules)
endif()

if (CATALYST_GTEST_AVAILABLE)
  list(APPEND DIALECT_TESTS_DEPEND CatalystUnitTests)
endif()

add_lit_testsuite(check-dialects "Run the regression tests for mlir dialects"
    ${TEST_SUITES}
    DEPENDS ${DIALECT_TESTS_DEPEND}
)

set_target_properties(check-dialects PROPERTIES FOLDER "Tests")

add_subdirectory(frontend)

add_custom_target(check-all
    DEPENDS
    check-dialects
    check-frontend
)


add_lit_testsuite(check-unit-tests "Just run unit tests"
    ${CMAKE_CURRENT_BINARY_DIR}/Unit
    DEPENDS ${DIALECT_TEST_DEPENDS} check-dialects
)

set_target_properties(check-unit-tests PROPERTIES FOLDER "Unit Tests")

