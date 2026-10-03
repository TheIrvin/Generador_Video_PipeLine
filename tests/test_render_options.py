import json

import pytest

from animavideo.render import RenderOptions


def test_default_options_are_valid():
    options = RenderOptions()

    options.validate()

    assert options.fps == 24
    assert options.width == 720
    assert options.height == 1280


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_seconds", "10"),
        ("duration_seconds", True),
        ("fps", "24"),
        ("fps", True),
        ("width", "720"),
        ("height", 720.0),
        ("zoom_start", "1.0"),
        ("zoom_end", False),
        ("focal_x", "0.5"),
        ("focal_y", None),
        ("hold_seconds", "0.5"),
        ("duration_seconds", float("nan")),
        ("zoom_start", float("inf")),
        ("focal_x", float("-inf")),
        ("preset", 1),
    ],
)
def test_from_json_rejects_invalid_option_types(field, value):
    with pytest.raises(ValueError, match=field):
        RenderOptions.from_json(json.dumps({field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_seconds", 0),
        ("duration_seconds", 31),
        ("fps", 0),
        ("fps", 61),
        ("width", 63),
        ("width", 1921),
        ("height", 63),
        ("height", 1921),
        ("width", 721),
        ("height", 721),
        ("zoom_start", 0.9),
        ("zoom_end", 4.1),
        ("focal_x", -0.1),
        ("focal_y", 1.1),
        ("hold_seconds", -0.1),
    ],
)
def test_from_json_rejects_out_of_range_options(field, value):
    with pytest.raises(ValueError, match=field):
        RenderOptions.from_json(json.dumps({field: value}))
