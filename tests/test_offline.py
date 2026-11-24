"""Software checks; these do not claim a hardware NPU inference result."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
from tools.detect_ssd import ANCHORS, decode


class SSDDecoderTests(unittest.TestCase):
    def test_known_boxes_class_mapping_and_duplicate_suppression(self):
        target = np.array([0.1, 0.2, 0.5, 0.8], dtype=np.float32)
        loc = np.zeros((8732, 4), dtype=np.float32)
        logits = np.zeros((8732, 81), dtype=np.float32)
        logits[:, 0] = 20  # Background everywhere except three selected anchors.
        for idx, class_id, score in [(0, 1, 30), (1, 1, 29), (2, 17, 28)]:
            anchor = ANCHORS[idx]
            anchor_size = anchor[2:] - anchor[:2]
            loc[idx, :2] = ((target[:2] + target[2:]) / 2 -
                           (anchor[:2] + anchor[2:]) / 2) / anchor_size / 0.1
            loc[idx, 2:] = np.log((target[2:] - target[:2]) / anchor_size) / 0.2
            logits[idx, class_id] = score
        results = decode(loc, logits, 100, 200, 0.3)
        self.assertEqual([r['label'] for r in results], ['person', 'dog'])
        np.testing.assert_allclose(results[0]['box'], [10, 40, 50, 160], atol=0.02)

    def test_background_is_empty(self):
        logits = np.zeros((8732, 81), dtype=np.float32)
        logits[:, 0] = 20
        self.assertEqual(decode(np.zeros((8732, 4)), logits, 100, 100, 0.3), [])


class OfflineWorkflowTests(unittest.TestCase):
    def run_isolated_case(self, result=None, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(task='检查人员并记录风险', image=str(agent.ROOT / 'tests/fixtures/blank.jpg'),
                                      observation='', npu_detect=True, detections=None,
                                      tool_error=False, confirm=None, confirm_after=False, llm=False)
            with patch.multiple(agent, DATA=root, DB=root/'knowledge.sqlite3',
                                REPORTS=root/'reports', EVIDENCE=root/'evidence'), \
                 patch.dict(agent.CONFIG, {'person_detection': {}}):
                with patch.object(agent, 'run_npu_detector', return_value=result, side_effect=failure), \
                     patch.object(agent, 'run_fire_smoke_detector', side_effect=RuntimeError('smoke tool disabled in test')):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        agent.run_case(args)
            record_path = next((root/'reports').glob('*.json'))
            record = json.loads(record_path.read_text())
            page = record_path.with_suffix('.html').read_text()
            self.assertTrue(Path(record['image']['path']).is_file())
            return record, page

    def test_detection_evidence_and_uncovered_fire_capability(self):
        detections = [{'label': 'person', 'confidence': .92, 'box': [10, 20, 100, 200]}]
        record, page = self.run_isolated_case((detections, {'image_size': [640, 427], 'detections': detections}))
        self.assertEqual(record['detections'], detections)
        self.assertEqual(record['detection_source'], 'cix_ssd_300_npu')
        self.assertEqual(record['risk_level'], '无法评估：检测未完成')
        self.assertIn('fire', record['vision_missing_capabilities'])
        self.assertIn('<rect ', page)
        self.assertIn('person #1 0.92', page)
        self.assertEqual(record['person_statistics']['estimated_count'], 1)
        self.assertIn('估计有 1 个人员目标', page)
        self.assertEqual(record['human_confirmation'], '未确认')

    def test_tool_failure_still_saves_report(self):
        record, page = self.run_isolated_case(failure=RuntimeError('simulated device failure'))
        self.assertTrue(record['tool_error'])
        self.assertEqual(record['risk_level'], '无法评估：检测未完成')
        self.assertIn('simulated device failure', record['tool_error_message'])
        self.assertIn('无法评估：检测未完成', page)
        self.assertIsNone(record['person_statistics']['estimated_count'])


class PersonCountingTests(unittest.TestCase):
    def record(self, detections, source='cix_ssd_300_npu', failed=False):
        return {'task': '检察人员情况，并统计有几个人', 'detections': detections,
                'detection_source': source, 'vision_result': {'threshold': 0.3}, 'tool_error': failed}

    def test_five_boxes_keep_uncertain_person_in_estimate(self):
        confidence = [.989869, .966193, .949982, .887106, .442249]
        record = self.record([{'label': 'person', 'confidence': v} for v in confidence] +
                             [{'label': 'chair', 'confidence': .9}])
        result = agent.with_task_result(record)
        self.assertEqual(result['task_kind'], 'person_count')
        self.assertEqual(result['person_statistics']['estimated_count'], 5)
        self.assertEqual(result['person_statistics']['high_confidence_count'], 4)
        self.assertEqual(result['person_statistics']['review_count'], 1)
        self.assertIn('估计有 5 个人员目标', result['task_answer'])
        self.assertNotIn('person_statistics', record)

    def test_success_with_no_boxes_differs_from_failure_or_disabled_detection(self):
        success = agent.with_task_result(self.record([]))
        self.assertEqual(success['person_statistics']['estimated_count'], 0)
        self.assertIn('不能据此认定现场无人', success['task_answer'])
        for source, failed in [('cix_ssd_300_npu_failed', True), ('none', False)]:
            record = agent.with_task_result(self.record([], source, failed))
            self.assertIsNone(record['person_statistics']['estimated_count'])
            self.assertIn('无法统计人数', record['task_answer'])

    def test_external_count_preserves_unverified_origin(self):
        result = agent.with_task_result(self.record([{'label': 'person', 'confidence': .9}],
                                                   'external_json_unverified'))
        self.assertIn('外部 JSON', result['task_answer'])
        self.assertIn('来源尚未验证', result['task_answer'])


class QuestionAnsweringTests(unittest.TestCase):
    def setUp(self):
        model_digest = patch.object(agent, 'model_digest', return_value='software-test-model-digest')
        model_digest.start()
        self.addCleanup(model_digest.stop)

    def test_question_routing_does_not_force_people_counts(self):
        for question in ['解释一下离线运行是什么意思', '写两句鼓励救援队的话', '怎么统计人数？']:
            plan = agent.request_plan(question, has_image=True)
            self.assertEqual(plan['kind'], 'general_question', question)
            self.assertFalse(plan['use_detection'])
            self.assertFalse(plan['show_people'])
            self.assertFalse(plan['show_risk'])
        self.assertEqual(agent.request_plan('这张图有多少个人？', True)['kind'], 'person_count')
        self.assertEqual(agent.request_plan('检查人员并记录风险', True)['kind'], 'person_inspection')
        self.assertTrue(agent.request_plan('检查人员并记录风险', True)['show_risk'])
        self.assertEqual(agent.request_plan('这些人在哪？', True)['kind'], 'person_inspection')
        self.assertEqual(agent.request_plan('这张图有烟雾吗？', True)['kind'], 'risk_assessment')

    def test_unsupported_image_question_is_not_replaced_by_counting(self):
        for task in ['这些人穿什么衣服？', '图片中有没有人受伤？']:
            record = agent.with_task_result({'task': task, 'image': {'path': 'image.jpg'},
                                            'detections': [], 'detection_source': 'none'})
            self.assertEqual(record['task_kind'], 'visual_question')
            self.assertIn('不能根据图片判断', record['task_answer'])
            self.assertNotIn('统计人数', record['task_answer'])

    def test_generated_multiline_answer_survives_save_and_history_enrichment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer = '第一句：加油。\n第二句：注意安全。'
            args = argparse.Namespace(task='写两句鼓励救援队的话', image=str(agent.ROOT/'tests/fixtures/blank.jpg'),
                                      observation='', npu_detect=True, detections=None, tool_error=False,
                                      confirm=None, confirm_after=False, llm=True)
            with patch.multiple(agent, DATA=root, DB=root/'knowledge.sqlite3',
                                REPORTS=root/'reports', EVIDENCE=root/'evidence'), \
                 patch.object(agent, 'explain', return_value=(answer, None)), \
                 patch.object(agent, 'run_npu_detector') as detector, \
                 contextlib.redirect_stdout(io.StringIO()):
                record = agent.run_case(args)
                detector.assert_not_called()  # Attached image does not force a vision task.
                self.assertEqual(record['task_answer'], answer)
                self.assertEqual(record['answer_source'], 'local_qwen')
                saved = json.loads((root/'reports'/f"{record['case_id']}.json").read_text())
                self.assertEqual(agent.with_task_result(saved)['task_answer'], answer)
                report = (root/'reports'/f"{record['case_id']}.md").read_text()
                self.assertIn(answer, report)
                self.assertNotIn('人员统计：', report)
                self.assertNotIn('风险提示：', report)

    def test_old_fixed_count_answer_is_not_reused_for_unrelated_question(self):
        record = agent.with_task_result({'task': '什么是离线运行？', 'task_answer': '本张图片估计有 5 人',
                                        'detections': [], 'detection_source': 'none'})
        self.assertEqual(record['task_kind'], 'general_question')
        self.assertNotIn('5 人', record['task_answer'])


if __name__ == '__main__':
    unittest.main()
