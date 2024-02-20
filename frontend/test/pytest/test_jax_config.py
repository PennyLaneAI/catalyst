import jax

from catalyst.utils.jax_extras import transient_jax_config


def test_transient_jax_config():
    """Test that the ``transient_jax_config()`` context manager updates and
    restores the value of the JAX dynamic shapes option.
    """
    assert jax.config.jax_dynamic_shapes is False  # type: ignore

    with transient_jax_config():
        assert jax.config.jax_dynamic_shapes is True  # type: ignore

    assert jax.config.jax_dynamic_shapes is False  # type: ignore
