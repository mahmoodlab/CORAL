"""Generic class-registry decorator factory.

CORAL has multiple ``{name -> class}`` registries — feature extractors,
tissue segmenters, and likely more in later sprints. Each used to
ship its own near-identical ``@register(name)`` decorator. This
module hosts a single factory: each registry just calls
:func:`make_register_decorator` with its own dict + ABC and gets the
right decorator back.

The decorator is typed to **preserve the concrete decorated class**
(via :class:`_ClassRegistrar`), so ``Concrete = register(...)(Concrete)``
keeps ``Concrete``'s own constructor signature rather than collapsing
to the registry's base class.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Protocol, TypeVar, cast

T = TypeVar("T")
C = TypeVar("C")


class _ClassRegistrar(Protocol):
    """A ``register(name)`` decorator that returns its class unchanged."""

    def __call__(self, cls: type[C], /) -> type[C]: ...


def _check_subclass(
    cls: type, base_class: type, *, kind: str, name: str
) -> None:
    """Raise ``TypeError`` unless ``cls`` subclasses ``base_class``."""
    if not issubclass(cls, base_class):
        msg = (
            f"Cannot register {cls.__name__!r} as {kind} {name!r}: "
            f"not a {base_class.__name__} subclass."
        )
        raise TypeError(msg)


def make_register_decorator(
    registry: dict[str, type[T]],
    *,
    kind: str,
    base_class: type[T],
) -> Callable[[str], _ClassRegistrar]:
    """Build a ``@register(name)`` decorator for a class registry.

    The returned decorator:

    - Validates that the registered class is a ``base_class``
      subclass; raises ``TypeError`` otherwise.
    - Emits a ``UserWarning`` on re-registration of an existing name
      and overwrites the prior entry.
    - Returns the class unchanged (with its concrete type preserved) so
      the decorator composes with others.

    Args:
        registry: Module-level dict that maps name → class.
        kind: Human-readable category name used in warnings + errors
            (e.g. ``"extractor"``, ``"segmenter"``).
        base_class: ABC the registered class must subclass.

    Returns:
        A ``register(name)`` decorator function.

    Example:
        >>> from abc import ABC
        >>> class _Base(ABC):
        ...     pass
        >>> _registry: dict[str, type[_Base]] = {}
        >>> reg = make_register_decorator(
        ...     _registry, kind="thing", base_class=_Base
        ... )
        >>> @reg("foo")
        ... class _Foo(_Base):
        ...     pass
        >>> _registry["foo"] is _Foo
        True
    """

    def register(name: str) -> _ClassRegistrar:
        def decorator(cls: type[C]) -> type[C]:
            _check_subclass(cls, base_class, kind=kind, name=name)
            if name in registry:
                warnings.warn(
                    f"Re-registering {kind} name {name!r}; "
                    f"replacing {registry[name].__name__} "
                    f"with {cls.__name__}.",
                    category=UserWarning,
                    stacklevel=2,
                )
            registry[name] = cast("type[T]", cls)
            return cls

        return decorator

    return register
