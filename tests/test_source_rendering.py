import asyncio
from uuid import uuid4

import pytest

from app.carebridge import rendering
from app.carebridge.rendering import clear_category, render_segment
from app.carebridge.schemas import DraftRequest, ModelSegment
from app.carebridge.service import DraftService
from app.carebridge.workflow import (
    EditRequest,
    GenerationRequest,
    generation_result,
    pipeline_info,
    prepare_edit,
)
from tests.conftest import FakeProvider


def segment(text, category="other", confirmation=True):
    return ModelSegment(
        segment_id="s001", translation=text, category=category, needs_confirmation=confirmation
    )


@pytest.mark.parametrize(
    "language,source,before,after",
    [
        (
            "vi",
            "Sáng nay không uống nước.",
            "오늘 아침 물을 마셨지 않았습니다.",
            "오늘 아침 물을 마시지 않았습니다.",
        ),
        (
            "en",
            'Said, "I did not drink milk."',
            '"우유를 마셨지 않았다"고 말했다.',
            '"우유를 마시지 않았다"고 말했다.',
        ),
        (
            "vi",
            'Nói rằng: "Tôi không uống sữa."',
            '말했다. "우유를 마셨지 않다." ',
            '"우유를 마시지 않았다."라고 말했습니다.',
        ),
        ("en", "Ate half of dinner.", "저녁 반 정도 먹었습니다.", "저녁 식사를 절반 먹었습니다."),
        (
            "id",
            "Sudah makan setengah sarapan.",
            "아침 식사 반량을 먹었습니다.",
            "아침 식사를 절반 먹었습니다.",
        ),
        (
            "en",
            "Fell asleep 23 minutes after lunch.",
            "점심 식사 후 23분 후에 잠들었음",
            "점심 식사 23분 후에 잠들었습니다.",
        ),
        (
            "id",
            "Pergi ke kamar dengan kursi roda dan bantuan.",
            "휠체어로 도움을 받아 방으로 갔습니다.",
            "도움을 받아 휠체어를 이용해 방으로 갔습니다.",
        ),
        (
            "id",
            "Mungkin sudah makan siang, tetapi belum dikonfirmasi.",
            "점심 식사했을 수도 있지만, 확인되지 않았습니다.",
            "이미 점심 식사를 했을 수도 있지만, 확인되지 않았습니다.",
        ),
    ],
)
def test_bounded_wording_preserves_facts_and_is_idempotent(language, source, before, after):
    original = segment(before)
    result = render_segment(source, language, original)
    assert result.translation == after
    assert original.translation == before
    assert render_segment(source, language, result) == result


@pytest.mark.parametrize(
    "language,source,target",
    [
        ("en", "Ate about half of dinner.", "저녁 반 정도 먹었습니다."),
        ("en", "Ate half of dinner, maybe.", "저녁 반 정도 먹었습니다."),
        ("en", "Ate half of dinner.", "점심 반 정도 먹었습니다."),
        ("vi", "Đã ăn khoảng một nửa bữa trưa.", "점심 반 정도 먹었습니다."),
        ("id", "Sudah makan sekitar setengah sarapan.", "아침 식사 반량을 먹었습니다."),
        ("en", "Slept for 23 minutes after lunch.", "점심 식사 후 23분 후에 잠들었음"),
        ("en", "Fell asleep 24 minutes after lunch.", "점심 식사 후 23분 후에 잠들었음"),
        ("en", "Fell asleep 23 hours after lunch.", "점심 식사 후 23분 후에 잠들었음"),
        ("id", "Pergi ke kamar dengan kursi roda.", "휠체어로 도움을 받아 방으로 갔습니다."),
        (
            "id",
            "Mungkin makan siang, tetapi belum dikonfirmasi.",
            "점심 식사했을 수도 있지만, 확인되지 않았습니다.",
        ),
        (
            "id",
            "Mungkin sudah makan siang, tetapi belum dikonfirmasi.",
            "저녁 식사했을 수도 있지만, 확인되지 않았습니다.",
        ),
        ("en", "Drank milk.", "우유를 마셨지 않았습니다."),
        ("auto", "Did not drink milk.", "우유를 마셨지 않았습니다."),
    ],
)
def test_does_not_rewrite_unsupported_or_conflicting_claims(language, source, target):
    assert render_segment(source, language, segment(target)).translation == target


@pytest.mark.parametrize(
    "language,source,category",
    [
        ("en", 'Said, "I did not drink juice."', "hydration"),
        ("en", 'Said, "I have never drunk milk."', "hydration"),
        ("vi", 'Nói rằng: "Tôi chưa bao giờ uống sữa."', "hydration"),
        ("id", 'Mengatakan, "Saya tidak makan bubur."', "meal"),
        ("id", 'Mengatakan, "Saya tidak keluar rumah hari ini."', "mobility"),
        ("vi", 'Nói rằng: "Hôm nay tôi không ra ngoài."', "mobility"),
        ("en", "At 13:45, had not taken the medicine yet.", "medication"),
        ("id", "Pada pukul 07:15 belum minum obat.", "medication"),
        ("id", "Tidak minum air pagi ini.", "hydration"),
        ("id", "Mengatakan tidak ada rasa sakit hari ini.", "condition"),
    ],
)
def test_clear_full_clause_metadata_is_about_the_reported_action(language, source, category):
    assert clear_category(source, language) == category
    result = render_segment(source, language, segment("내용을 그대로 번역했습니다."))
    assert result.category == category
    assert result.needs_confirmation is False
    assert result.translation == "내용을 그대로 번역했습니다."


@pytest.mark.parametrize(
    "language,source",
    [
        ("en", 'Said, "I may not have drunk juice."'),
        ("vi", 'Nói rằng: "Tôi có thể đã uống sữa."'),
        ("id", 'Mengatakan, "Saya mungkin tidak minum susu."'),
        ("en", 'Said, "I did not drink juice." but the glass was empty.'),
        ("en", 'Said, "I did not drink juice or eat soup."'),
        ("en", 'Said, "I did not drink it."'),
        ("en", 'Said, "I did not drink medicine."'),
        ("id", 'Mengatakan, "Saya tidak minum obat."'),
        ("id", "Tidak minum air pagi ini?"),
        ("en", "At 25:80, had not taken the medicine yet."),
        ("en", "At 13:45, had not taken the medicine yet, possibly."),
        ("id", "Pada pukul 07:15 belum minum obat yang tidak diketahui."),
    ],
)
def test_unknown_ambiguous_uncertain_and_mixed_clauses_keep_model_flags(language, source):
    assert clear_category(source, language) is None
    original = segment("확인이 필요합니다.", "medication")
    assert render_segment(source, language, original) == original


def test_rendered_service_keeps_source_evidence_and_normal_safety_checks():
    source = 'Said, "I did not drink milk."'
    provider = FakeProvider(['"우유를 마셨지 않았다"고 말했다.'], needs_confirmation=True)
    request = DraftRequest(memo=source, source_language="en", data_use="synthetic_deidentified")
    draft = asyncio.run(DraftService(provider).create(request))
    assert draft.translation.text == '"우유를 마시지 않았다"고 말했다.'
    assert draft.observations[0].category == "hydration"
    assert not draft.additional_checks
    assert draft.translation.segments[0].evidence.quote == source
    assert draft.source.text == source
    assert draft.qa.requires_human_review and not draft.qa.external_share_allowed
    # Metadata clarification must never pass a translation with lost negation.
    bad = asyncio.run(DraftService(FakeProvider(["우유를 마셨다고 말했습니다."])).create(request))
    assert bad.status == "hold"


def test_uncertainty_check_survives_wording_changes():
    request = DraftRequest(
        memo="Mungkin sudah makan malam, tetapi belum dikonfirmasi.",
        source_language="id",
        data_use="synthetic_deidentified",
    )
    draft = asyncio.run(
        DraftService(FakeProvider(["저녁 식사했을 수도 있지만, 확인되지 않았습니다."])).create(
            request
        )
    )
    assert draft.translation.text.startswith("이미")
    assert len(draft.additional_checks) == 1


def test_korean_source_and_human_edits_are_not_rewritten(config):
    original = "오늘 아침 물을 마셨지 않았습니다."
    korean = DraftRequest(memo=original, source_language="ko", data_use="synthetic_deidentified")
    assert asyncio.run(DraftService(FakeProvider()).create(korean)).translation.text == original
    request = GenerationRequest(
        memo="Did not drink milk.",
        source_language="en",
        data_use="synthetic_deidentified",
        request_id=str(uuid4()),
        source_revision=1,
    )
    draft = asyncio.run(DraftService(FakeProvider(["우유를 마시지 않았습니다."])).create(request))
    generation = generation_result(request, draft, config)
    edited = EditRequest(
        data_use="synthetic_deidentified",
        request_id=str(uuid4()),
        generation=generation,
        expected_source_revision=1,
        expected_source_sha256=draft.source.sha256,
        editor="author",
        segments=[{"segment_id": "s001", "text": original, "category": "hydration"}],
    )
    result, _ = prepare_edit(edited)
    assert result.translation.text == original


def test_rendering_rules_affect_pipeline_fingerprint(config, monkeypatch):
    before = pipeline_info(config).translation_prompt_sha256
    monkeypatch.setattr(rendering, "PAST_NEGATION", "never-match-this-rule")
    assert pipeline_info(config).translation_prompt_sha256 != before


def test_presentation_expansion_never_breaks_the_output_length_contract():
    prefix, suffix = '말했다. "', '"'
    original = prefix + "가" * (4000 - len(prefix) - len(suffix)) + suffix
    result = render_segment('Said, "I did not drink milk."', "en", segment(original))
    assert result.translation == original
