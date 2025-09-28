{{ fullname | escape | underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}


{% if classes %}
.. rubric:: Classes

.. autosummary::
    :recursive:
    :toctree: .
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions 

.. autosummary::
    :recursive:
    :toctree: .
    {% for function in functions %}
    .. autofunction:: {{function}}
    {% endfor %}

{% endif %}