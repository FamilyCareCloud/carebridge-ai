"""Uncertainty stays visible; absent location context is never invented."""

import asyncio

import pytest

from app.carebridge.rendering import render_segment
from app.carebridge.schemas import DraftRequest
from app.carebridge.service import DraftService
from tests.conftest import FakeProvider
from tests.test_source_rendering import segment


@pytest.mark.parametrize(
    "source_meal,target_meal", [("sáng", "아침"), ("trưa", "점심"), ("tối", "저녁")]
)
def test_vietnamese_completed_possibility_is_still_unconfirmed(source_meal, target_meal):
    source = f"Có thể đã ăn {source_meal}, nhưng chưa được xác nhận."
    original = segment(
        f"{target_meal} 식사했을 수도 있지만, 확인되지 않았습니다.", confirmation=False
    )
    result = render_segment(source, "vi", original)
    assert result.translation == f"{target_meal} 식사를 했을 수도 있지만, 아직 확인되지 않았습니다."
    assert result.needs_confirmation is True
    assert result.category == original.category
    assert render_segment(source, "vi", result) == result


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "Có thể sẽ ăn tối, nhưng chưa được xác nhận.",
            "저녁 식사했을 수도 있지만, 확인되지 않았습니다.",
        ),
        (
            "Có thể ăn tối, nhưng chưa được xác nhận.",
            "저녁 식사했을 수도 있지만, 확인되지 않았습니다.",
        ),
        (
            "Có thể đã ăn tối, nhưng chưa được xác nhận. Đã uống nước.",
            "저녁 식사했을 수도 있지만, 확인되지 않았습니다.",
        ),
        (
            "Có thể đã ăn trưa, nhưng chưa được xác nhận.",
            "저녁 식사했을 수도 있지만, 확인되지 않았습니다.",
        ),
        ("Có thể đã ăn tối, nhưng chưa được xác nhận.", "이미 저녁을 먹었고 확인되었습니다."),
        (
            "Có thể đã ăn tối, nhưng chưa được xác nhận.",
            "저녁 식사했을 수도 있지만, 확인되지 않았습니다. 물도 마셨습니다.",
        ),
    ],
)
def test_unknown_meal_clauses_and_mismatches_abstain(source, target):
    original = segment(target, confirmation=False)
    assert render_segment(source, "vi", original) == original


@pytest.mark.parametrize("target_place", ["정원", "공원"])
def test_bare_taman_cannot_silently_choose_a_location(target_place):
    source = "Tidak berjalan ke taman; pergi ke sana dengan kursi roda dan bantuan."
    original = segment(
        f"{target_place}까지 걸어서 이동하지 않았으며, 도움을 받아 휠체어를 이용해 갔습니다.",
        confirmation=False,
    )
    result = render_segment(source, "id", original)
    assert result.translation.startswith("정원 또는 공원까지")
    assert result.needs_confirmation is True
    assert render_segment(source, "id", result) == result


@pytest.mark.parametrize(
    "source_place,target_place",
    [("kamar", "방"), ("ruang makan", "식사실"), ("taman kota", "공원"), ("taman rumah", "정원")],
)
def test_qualified_and_unambiguous_locations_keep_their_meaning(source_place, target_place):
    source = f"Tidak berjalan ke {source_place}; pergi ke sana dengan kursi roda dan bantuan."
    original = segment(
        f"{target_place}까지 걸어서 이동하지 않았으며, 도움을 받아 휠체어를 이용해 갔습니다.",
        confirmation=False,
    )
    assert render_segment(source, "id", original) == original


def test_location_ambiguity_reaches_the_existing_source_author_check():
    source = "Tidak berjalan ke taman; pergi ke sana dengan kursi roda dan bantuan."
    draft = asyncio.run(
        DraftService(
            FakeProvider(["공원에 가는 도중 걸어 가지 않았으며 휠체어와 도움을 받아 갔습니다."])
        ).create(DraftRequest(memo=source, source_language="id", data_use="synthetic_deidentified"))
    )
    assert draft.translation.text.startswith("정원 또는 공원까지")
    assert len(draft.additional_checks) == 1
    assert draft.additional_checks[0].evidence.quote == source
    assert draft.qa.external_share_allowed is False


@pytest.mark.parametrize(
    "source_meal,target_meal", [("sáng", "아침"), ("trưa", "점심"), ("tối", "저녁")]
)
@pytest.mark.parametrize(
    "wording",
    [
        "{meal} 식사했을 수도 있지만 확인되지 않았습니다.",
        "{meal}을 먹었을 수도 있지만, 확인되지 않았습니다.",
    ],
)
def test_unconfirmed_meal_wording_variants(source_meal, target_meal, wording):
    source = f"Có thể đã ăn {source_meal}, nhưng chưa được xác nhận."
    result = render_segment(source, "vi", segment(wording.format(meal=target_meal)))
    assert result.translation == f"{target_meal} 식사를 했을 수도 있지만, 아직 확인되지 않았습니다."


@pytest.mark.parametrize(
    "place,target_place,before",
    [
        ("taman kota", "도시 공원", "도움을 받아 휠체어로 공원에 갔습니다."),
        ("taman rumah", "집 정원", "정원으로 걸어 가지 않았으며, 휠체어와 도움을 받아 갔습니다."),
    ],
)
def test_explicit_place_and_walking_negation_survive_qualified_transport(
    place, target_place, before
):
    source = f"Tidak berjalan ke {place}; pergi ke sana dengan kursi roda dan bantuan."
    result = render_segment(source, "id", segment(before, confirmation=False))
    assert (
        result.translation
        == f"{target_place}까지 걸어서 이동하지 않았으며, 도움을 받아 휠체어를 이용해 갔습니다."
    )
    assert result.needs_confirmation is False
    assert render_segment(source, "id", result) == result


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "Tidak berjalan ke taman kota; pergi ke sana dengan kursi roda dan bantuan. Jatuh.",
            "도움을 받아 휠체어로 공원에 갔습니다.",
        ),
        (
            "Berjalan ke taman kota; pergi ke sana dengan kursi roda dan bantuan.",
            "도움을 받아 휠체어로 공원에 갔습니다.",
        ),
        (
            "Tidak berjalan ke taman kota; pergi ke sana dengan kursi roda dan bantuan.",
            "도움을 받아 휠체어로 정원에 갔습니다.",
        ),
        (
            "Tidak berjalan ke taman kota; pergi ke sana dengan kursi roda dan bantuan.",
            "도움을 받아 휠체어로 공원에 갔습니다. 넘어졌습니다.",
        ),
    ],
)
def test_qualified_transport_abstains_from_extra_facts_and_place_mismatch(source, target):
    original = segment(target, confirmation=False)
    assert render_segment(source, "id", original) == original


def test_qualified_transport_still_runs_negation_and_assistance_qa():
    source = "Tidak berjalan ke taman kota; pergi ke sana dengan kursi roda dan bantuan."
    draft = asyncio.run(
        DraftService(FakeProvider(["도움을 받아 휠체어로 공원에 갔습니다."])).create(
            DraftRequest(memo=source, source_language="id", data_use="synthetic_deidentified")
        )
    )
    assert draft.status == "review_required"
    assert all(c.passed for c in draft.qa.checks)
    assert not draft.additional_checks
