from ..ir import *

class QNodeOp:
    @property
    def name(self) -> StringAttr:
        return StringAttr(self.attributes["sym_name"])
