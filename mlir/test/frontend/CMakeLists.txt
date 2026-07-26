set(FRONTEND_LIT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../frontend/test/lit")
set(DECOMP_RULES_BYTECODE_PATH "${FRONTEND_LIT_TEST_DIR}/GraphDecomposition/test_rules.mlirbc")

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${FRONTEND_LIT_TEST_DIR}/lit.cfg.py
)

add_custom_target(create_test_bytecode
    quantum-opt --emit-bytecode --register-decomp-rule-resource ${FRONTEND_LIT_TEST_DIR}/GraphDecomposition/test_rules.mlir -o ${DECOMP_RULES_BYTECODE_PATH}
    DEPENDS quantum-opt
    BYPRODUCTS ${DECOMP_RULES_BYTECODE_PATH}
)

add_lit_testsuite(check-frontend "Run the frontend tests"
    .  # the frontend tests are located in the same directory as the config file
    DEPENDS catalyst-cli create_test_bytecode
)
