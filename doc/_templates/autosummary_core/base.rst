{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ mod }}.{{ objname }}
={% for i in range(mod|length) %}={% endfor %}{{ underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
