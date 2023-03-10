#!/bin/sh

set -e -x

TAG=628f56

for f in _img/*$TAG.svg; do
  LANG=C inkscape -w 1024 -h 1024 "$f" -o "$f.png"
done

cat >_preview.md <<EOF
Regular circuits
================

![](./_img/regular_compile_$TAG.svg.png)

![](./_img/regular_compile_trial_$TAG.svg.png)

![](./_img/regular_runtime_$TAG.svg.png)

![](./_img/regular_runtime_trial_$TAG.svg.png)

Deep circuits
=============

![](./_img/deep_compile_$TAG.svg.png)

![](./_img/deep_compile_trial_$TAG.svg.png)

![](./_img/deep_runtime_$TAG.svg.png)

![](./_img/deep_runtime_trial_$TAG.svg.png)

Variational circuits (line plots)
=================================

![](./_img/variational_compile_adjoint_backprop_lineplot_$TAG.svg.png)

![](./_img/variational_compile_trial_adjoint_backprop_lineplot_$TAG.svg.png)

![](./_img/variational_compile_finitediff_lineplot_$TAG.svg.png)

![](./_img/variational_compile_trial_finitediff_lineplot_$TAG.svg.png)

![](./_img/variational_compile_parametershift_lineplot_$TAG.svg.png)

![](./_img/variational_compile_trial_parametershift_lineplot_$TAG.svg.png)


![](./_img/variational_runtime_adjoint_backprop_lineplot_$TAG.svg.png)

![](./_img/variational_runtime_trial_adjoint_backprop_lineplot_$TAG.svg.png)

![](./_img/variational_runtime_finitediff_lineplot_$TAG.svg.png)

![](./_img/variational_runtime_trial_finitediff_lineplot_$TAG.svg.png)

![](./_img/variational_runtime_parametershift_lineplot_$TAG.svg.png)

![](./_img/variational_runtime_trial_parametershift_lineplot_$TAG.svg.png)

Variational circuits (bar charts)
=================================

![](./_img/variational_runtime_adjoint_backprop_$TAG.svg.png)

![](./_img/variational_runtime_finitediff_$TAG.svg.png)

![](./_img/variational_runtime_parametershift_$TAG.svg.png)
EOF

pandoc _preview.md -o catalyst_benchmark_$TAG.pdf
