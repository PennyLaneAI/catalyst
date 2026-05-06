# Release 0.16.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

* Update RC nightly builds to read version number from the `_version.py` file 
  [(#2797)](https://github.com/PennyLaneAI/catalyst/pull/2797)

* Fix build failures when using clang with GCC ≤ 13 libstdc++ by replacing
  `std::views::filter`/`std::views::transform` with `std::copy_if`/`std::transform`
  [(#2801)](https://github.com/PennyLaneAI/catalyst/pull/2801)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Mehrdad Malekmohammadi,
