#!/bin/sh

set -e -x

TAG=628f56

for f in _img/*$TAG.svg; do
  LANG=C inkscape -w 1024 -h 1024 "$f" -o "$f.png"
done

cat >_preview.md <<EOF
Comments
========

1. We have 3 types of implementations (\`catalyst/lighting\`, \`pennylane/*\`, \`pennylane+jax/*\`) and we
   measure two values: \`compilation\` and \`running\` times. Totally we use 6 different measurement procedures
   encoded in the [catalyst_benchmark.main](https://github.com/XanaduAI/pennylane-mlir/blob/benchmarking-1-2/benchmark/catalyst_benchmark/main.py#L84)
   Python module.

2. For the data currently available, we followed the upstream terminology, e.g. we define
   \`PennyLane+Jax compilation\` as calling the \`compile\` function of the JAX function object.
   - The only exception (and a subject to change) is the \`PennyLane\` "running time". We let
     PennyLane do one warmup run and then define "running time" as running time of the
     subsequent runs.

3. For all the measurements we aim to put all the available data on the plots.

4. There are issues to be fixed regarding the plot title clipping.

\\newpage
Regular circuits
================

![](./_img/regular_compile_$TAG.svg.png)

![](./_img/regular_compile_trial_$TAG.svg.png)

![](./_img/regular_runtime_$TAG.svg.png)

![](./_img/regular_runtime_trial_$TAG.svg.png)

\\newpage
Deep circuits
=============

![](./_img/deep_compile_$TAG.svg.png)

![](./_img/deep_compile_trial_$TAG.svg.png)

![](./_img/deep_runtime_$TAG.svg.png)

![](./_img/deep_runtime_trial_$TAG.svg.png)

\\newpage
Variational circuits (line plots)
=================================

Comments:
- PLJax/def probably allows both \`adjoin\` and \`backprop\` methods, thus the collision

![](./_img/variational_compile_adjoint_backprop_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_compile_trial_adjoint_backprop_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_compile_finitediff_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_compile_trial_finitediff_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_compile_parametershift_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_compile_trial_parametershift_lineplot_$TAG.svg.png)


\\newpage
![](./_img/variational_runtime_adjoint_backprop_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_trial_adjoint_backprop_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_finitediff_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_trial_finitediff_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_parametershift_lineplot_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_trial_parametershift_lineplot_$TAG.svg.png)

\\newpage
Variational circuits (bar charts)
=================================

\\newpage
![](./_img/variational_runtime_adjoint_backprop_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_finitediff_$TAG.svg.png)

\\newpage
![](./_img/variational_runtime_parametershift_$TAG.svg.png)
EOF

pandoc \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=red \
    -V toccolor=gray \
    _preview.md -o catalyst_benchmark_$TAG.pdf
