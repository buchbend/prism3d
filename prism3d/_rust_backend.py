"""
Runtime detection and dispatch for the optional Rust backend.

The Rust extension module is compiled via maturin from rust/src/.
When available, it provides accelerated versions of the thermal solver,
chemistry integrator, PE table lookup, and RT cumsum.

The Python implementations remain the default fallback.
"""

_rust_available = None
_rust_module = None


def has_rust():
    """Check if the Rust extension module is available."""
    global _rust_available
    if _rust_available is None:
        try:
            import prism3d._prism3d_core  # noqa: F401
            _rust_available = True
        except ImportError:
            _rust_available = False
    return _rust_available


def get_rust():
    """Get the Rust extension module. Raises ImportError if not available."""
    global _rust_module
    if _rust_module is None:
        import prism3d._prism3d_core as mod
        _rust_module = mod
    return _rust_module
