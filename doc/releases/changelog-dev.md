# Release 0.14.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* A new option ``use_nameloc`` has been added to :func:`~.qjit` that embeds variable names
  from Python into the compiler IR, which can make it easier to read when debugging programs.
  [(#2054)](https://github.com/PennyLaneAI/catalyst/pull/2054)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixes the translation of plxpr control flow for edge cases where the `consts` were being
  reordered.
  [(#2128)](https://github.com/PennyLaneAI/catalyst/pull/2128)
  [(#2133)](https://github.com/PennyLaneAI/catalyst/pull/2133)

<h3>Internal changes ⚙️</h3>

* Refactor Catalyst pass registering so that it's no longer necessary to manually add new
  passes at `registerAllCatalystPasses`.
  [(#1984)](https://github.com/PennyLaneAI/catalyst/pull/1984)

* Split `from_plxpr.py` into two files.
  [(#2142)](https://github.com/PennyLaneAI/catalyst/pull/2142)

* xDSL passes are now automatically detected when using the `qjit` decorator. 
  This removes the need to pass the `pass_plugins` argument to the `qjit` decorator.
  [(#2169)](https://github.com/PennyLaneAI/catalyst/pull/2169)

<h3>Documentation 📝</h3>

* A typo in the code example for :func:`~.passes.ppr_to_ppm` has been corrected.
  [(#2136)](https://github.com/PennyLaneAI/catalyst/pull/2136)

* Fix `catalyst.qjit` and `catalyst.CompileOptions` docs rendering.
  [(#2156)](https://github.com/PennyLaneAI/catalyst/pull/2156)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee,
Roberto Turrado,
Paul Haochen Wang.
