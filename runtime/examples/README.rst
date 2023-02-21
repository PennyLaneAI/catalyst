The LLVM-QIR Examples
#####################

This directory includes several examples that showcase the QIR specification and runtime CAPIs,
which are implemented at the LLVM-QIR IR level. These examples demonstrate how to use QIR
to write quantum algorithms at the lowest IR level. You can modify the examples or create your
own based on the QIR specification and the runtime CAPIs.

To run the examples, you can use the Make targets provided in the Makefile. The Makefile contains
instructions to link the examples with the Catalyst runtime shared libraries and execute them.
To use these commands, you need to first build the Catalyst runtime shared libraries in the default
build subdirectory. The runtime provides the necessary support for executing quantum programs on the Lightning simulator.
