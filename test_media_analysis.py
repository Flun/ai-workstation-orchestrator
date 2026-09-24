import unittest

import media_analysis as ma
from media_analysis_backends import aligned_items_to_segments, restore_transcript_spacing


class MediaAnalysisHelpersTest(unittest.TestCase):
    def test_korean_alignment_becomes_timestamped_segments(self):
        segments = aligned_items_to_segments([
            {"word": "안녕하세요.", "start": 0.82, "end": 3.41},
            {"word": "오늘은", "start": 3.45, "end": 4.2},
            {"word": "테스트입니다.", "start": 4.21, "end": 7.92},
        ], "Korean")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "안녕하세요.")
        self.assertEqual(segments[1]["words"][0]["start"], 3.45)

    def test_srt_and_vtt_are_utf8_ready(self):
        segments = [{"start": 0.82, "end": 3.41, "text": "안녕하세요."}]
        self.assertIn("00:00:00,820 --> 00:00:03,410", ma._subtitle(segments, "srt"))
        self.assertTrue(ma._subtitle(segments, "vtt").startswith("WEBVTT"))

    def test_qwen_aligner_spacing_uses_asr_source_text(self):
        segments = [{"text": "인터넷 으로 믿음 이", "words": [
            {"word": "인터넷"}, {"word": "으로"}, {"word": "믿음"}, {"word": "이"},
        ]}]
        restored = restore_transcript_spacing(segments, "인터넷으로 믿음이")
        self.assertEqual(restored[0]["text"], "인터넷으로 믿음이")

    def test_combined_timeline_keeps_asr_as_source(self):
        asr = {"segments": [{"start": 1.0, "end": 4.0, "text": "정확한 대사"}]}
        caption = {"events": [{"start": 0.0, "end": 5.0, "description": "팬 소리가 들린다"}]}
        timeline = ma._merge_timeline(asr, caption, None)
        self.assertEqual(timeline[0]["speech"], "정확한 대사")
        self.assertEqual(timeline[0]["audio_description"], "팬 소리가 들린다")

    def test_chunk_transcripts_remove_boundary_overlap(self):
        text = ma._join_transcript_parts([
            "첫 번째 구간 마지막 문장",
            "마지막 문장 다음 구간입니다",
        ])
        self.assertEqual(text, "첫 번째 구간 마지막 문장 다음 구간입니다")

    def test_chunk_size_is_window_not_file_limit(self):
        payload = ma._payload(
            "detailed", "qwen3-asr", "ko", "transcribe", True, True,
            True, 5, 60, 2, "", True,
        )
        self.assertEqual(payload["chunk_size"], 60)
        self.assertEqual(payload["asr_chunk_size"], 60)


if __name__ == "__main__":
    unittest.main()
