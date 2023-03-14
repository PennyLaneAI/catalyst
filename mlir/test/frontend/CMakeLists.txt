configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_SOURCE_DIR}/../frontend/test/lit/lit.cfg.py
)

add_lit_testsuite(check-frontend "Run the frontend tests"
    .  # the frontend tests are located in the same directory as the config file
)
