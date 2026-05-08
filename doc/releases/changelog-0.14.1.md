# Release 0.14.1

<h3>Bug fixes</h3>

* The ``gast`` package is now an explicit dependency in Catalyst. The ``gast`` package was previously
  pulled in transitively by ``diastatic-malt``, but ``diastatic-malt==2.15.3`` dropped ``gast`` as
  a dependency, which caused an error when importing Catalyst.
  [#2565](https://github.com/PennyLaneAI/catalyst/pull/2565)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Ittah,
Haochen Paul Wang.
