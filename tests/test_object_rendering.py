"""Source-bound object correction must leave ambiguity and human edits alone."""

import pytest

from app.carebridge.rendering import render_segment
from tests.test_source_rendering import segment


@pytest.mark.parametrize(
    "language,source",
    [
        ("vi", 'Nói rằng: "Tôi có thể đã uống nước ép."'),
        ("en", 'Said, "I may have drunk juice."'),
        ("id", 'Mengatakan, "Saya mungkin sudah minum jus."'),
    ],
)
def test_quoted_juice_keeps_uncertainty_and_attribution(language, source):
    original = segment('"수프를 마셨을 수도 있습니다."라고 말했습니다.')
    result = render_segment(source, language, original)
    assert result.translation == '"주스를 마셨을 수도 있습니다."라고 말했습니다.'
    assert result.needs_confirmation is True
    assert result.category == original.category
    assert render_segment(source, language, result) == result
    assert original.translation.startswith('"수프')


@pytest.mark.parametrize(
    "source,target",
    [
        ('Nói rằng: "Tôi có thể đã uống súp."', '"수프를 마셨을 수도 있습니다."라고 말했습니다.'),
        (
            'Nói rằng: "Tôi có thể đã uống nước ép và súp."',
            '"수프를 마셨을 수도 있습니다."라고 말했습니다.',
        ),
        (
            'Nói rằng: "Tôi có thể đã uống nước ép."',
            '"수프를 두 잔 마셨을 수도 있습니다."라고 말했습니다.',
        ),
        (
            'Nói rằng: "Tôi có thể đã uống nước ép."',
            '"수프를 마셨을 수도 있습니다."라고 말했습니다. 물도 마셨습니다.',
        ),
        (
            'Nói rằng: "Tôi có thể đã uống nước ép táo."',
            '"수프를 마셨을 수도 있습니다."라고 말했습니다.',
        ),
        ("주스를 마셨을 수도 있다고 말했습니다.", '"수프를 마셨을 수도 있습니다."라고 말했습니다.'),
    ],
)
def test_unknown_or_compound_claims_are_not_rewritten(source, target):
    assert render_segment(source, "vi", segment(target)).translation == target


def test_transport_contrast_preserves_the_model_place_and_support():
    source = "Tidak berjalan ke taman; pergi ke sana dengan kursi roda dan bantuan."
    result = render_segment(
        source, "id", segment("공원에 가는 도중 걸어 가지 않았으며 휠체어와 도움을 받아 갔습니다.")
    )
    assert (
        result.translation
        == "공원까지 걸어서 이동하지 않았으며, 도움을 받아 휠체어를 이용해 갔습니다."
    )
    assert (
        render_segment(
            source.replace("taman", "kamar"), "id", segment(result.translation)
        ).translation
        == result.translation
    )
