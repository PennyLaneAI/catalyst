// This file ensures we are compiling with the patched verbs.h instead of the stock version.

#include <infiniband/verbs.h>

using VendoredVerbsHeaderInUse = decltype(ibv_context_ops::_compat_reg_mr_ex);
