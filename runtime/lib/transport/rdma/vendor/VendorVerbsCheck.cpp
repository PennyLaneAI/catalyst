// Compiled into every target that MUST see the patched infiniband/verbs.h in
// ../vendor rather than the system copy. It exists to make that a build error
// instead of a silent divergence.
//
// The vendored header adds one member to struct ibv_context_ops (`_compat_reg_mr_ex`,
// right after `_compat_reg_mr`). That struct is embedded in struct ibv_context by
// value, so compiling against the stock header instead shifts every later ops slot,
// and every ibv_context field after ops, by 8 bytes. It compiles cleanly and fails at
// run time by dispatching verbs calls through the wrong slots — which is precisely why
// the divergence has to be caught at build time.
//
// The include order that makes the vendored copy win is not self-enforcing. The vendor
// directory is passed as -isystem, and a compiler searches every -I directory before
// any -isystem one, so -isystem does not outrank a plain -I. The vendored copy wins
// only because no -I directory on these targets contains an infiniband/ subdirectory.
// That is an invariant someone can break without noticing; this file is what notices.

#include <infiniband/verbs.h>

// Naming the patched member IS the check. Against the stock header this fails to
// compile with "no member named '_compat_reg_mr_ex' in 'ibv_context_ops'".
// See rdma/vendor/README.md.
//
// Expect your editor to flag exactly that on the line below. This file only appears in
// compile_commands.json when the tree is configured with ENABLE_TRANSPORT_FPGA=ON, so
// without that a language server resolves <infiniband/verbs.h> to the system copy and
// reports the very error this check is designed to produce. It is not a defect.
using VendoredVerbsHeaderInUse = decltype(ibv_context_ops::_compat_reg_mr_ex);
