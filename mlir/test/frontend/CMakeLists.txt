set(BYTECODE_PATH "${CMAKE_SOURCE_DIR}/../frontend/test/lit/GraphDecomposition/test_rules.mlirbc")

add_custom_target(create_test_bytecode
    quantum-opt --emit-bytecode --register-decomp-rule-resource ${CMAKE_SOURCE_DIR}/../frontend/test/lit/GraphDecomposition/test_rules.mlir -o ${BYTECODE_PATH}
    DEPENDS quantum-opt
    BYPRODUCTS ${BYTECODE_PATH}
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_SOURCE_DIR}/../frontend/test/lit/lit.cfg.py
)

add_lit_testsuite(check-frontend "Run the frontend tests"
    .  # the frontend tests are located in the same directory as the config file
    DEPENDS
    create_test_bytecode
)
