"builtin.module"() ( {
  %0:4 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16, i32, i64)
  "dialect.op2"() ( {
    "dialect.innerop1"(%0#0, %0#1) : (i1, i16) -> ()
  },  {
    "dialect.innerop2"() : () -> ()
    "dialect.innerop3"(%0#0, %0#2, %0#3)[^bb1, ^bb2] : (i1, i32, i64) -> ()
  ^bb1(%1: i32):  // pred: ^bb0
    "dialect.innerop4"() : () -> ()
    "dialect.innerop5"() : () -> ()
  ^bb2(%2: i64):  // pred: ^bb0
    "dialect.innerop6"() : () -> ()
    "dialect.innerop7"() : () -> ()
  }) {"other attribute" = 42 : i64} : () -> ()
}) : () -> ()
