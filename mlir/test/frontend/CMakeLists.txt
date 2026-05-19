set(GRAPH_DECOMP_TEST_DIR "${CMAKE_SOURCE_DIR}/../frontend/test/lit/GraphDecomposition" CACHE PATH "Path to the graph-decomposition pass testing directory")

# used by frontend lit tests
set(BYTECODE_PATH "${GRAPH_DECOMP_TEST_DIR}/test_rules.mlirbc")

add_custom_command(
    OUTPUT "${BYTECODE_PATH}"
    COMMAND quantum-opt --emit-bytecode --register-decomp-rule-resource "${GRAPH_DECOMP_TEST_DIR}/test_rules.mlir" -o "${BYTECODE_PATH}"
    DEPENDS quantum-opt "${GRAPH_DECOMP_TEST_DIR}/test_rules.mlir"
)

add_custom_target(create_test_bytecode
    DEPENDS "${BYTECODE_PATH}"
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_SOURCE_DIR}/../frontend/test/lit/lit.cfg.py
)

add_lit_testsuite(check-frontend "Run the frontend tests"
    .  # the frontend tests are located in the same directory as the config file
    DEPENDS create_test_bytecode
)
