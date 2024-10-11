add_custom_target(CatalystUnitTests)
set_target_properties(CatalystUnitTests PROPERTIES FOLDER "Catalyst Tests")

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_catalyst_unittest test_dirname)
  add_unittest(CatalystUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Example)
