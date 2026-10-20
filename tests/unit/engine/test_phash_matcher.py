import pytest

from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.reference import ReferenceData

from ..sync_fakes import jpeg


@pytest.mark.unit
class TestBuildIndex:
    def test_unhashable_image_is_reported_and_left_out(self, tmp_path) -> None:
        good, truncated = tmp_path / "good.jpg", tmp_path / "truncated.jpg"
        good.write_bytes(jpeg(0))
        truncated.write_bytes(jpeg(1)[:200])
        matcher = PHashCardMatcher(reference=ReferenceData(tmp_path / "reference"))

        failures = matcher.build_index(
            {"uuid-good": good, "uuid-truncated": truncated}, num_workers=1
        )

        assert list(failures) == ["uuid-truncated"]
        assert matcher.uuid_mapping == ["uuid-good"]
        reloaded = PHashCardMatcher(reference=ReferenceData(tmp_path / "reference"))
        assert reloaded.uuid_mapping == ["uuid-good"]
