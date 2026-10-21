from laika import __version__


def test_package_version_is_pre_1_0() -> None:
    major = int(__version__.split(".", maxsplit=1)[0])
    assert major == 0
