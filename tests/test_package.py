import astro_weights


def test_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert astro_weights.__version__ == version("astro-weights")
