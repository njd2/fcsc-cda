from typing import TypeVar, Callable

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


def lookup_maybe(k: X, xs: dict[X, Y]) -> Y | None:
    return xs[k] if k in xs else None


def fmap_maybe(f: Callable[[X], Y], x: X | None) -> Y | None:
    return f(x) if x is not None else None


def lookup_map(f: Callable[[Y], Z | None], k: X, xs: dict[X, Y]) -> Z | None:
    return fmap_maybe(f, lookup_maybe(k, xs))


def float_maybe(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None
