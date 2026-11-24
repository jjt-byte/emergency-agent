"""Software checks for smoke/fire normalization and independent tool failures."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
from tools import detect_fire_smoke as detector
from tools import setup_fire_smoke as setup


class FireSmokeAdapterTests(unittest.TestCase):
    def test_background_excluded_and_coordinates_restored_to_original_image(self):
        output = {'class_ids': ['background', 'smoke', 'fire', 'fire'],
                  'scores': [.99, .8, .95, .2],
                  'boxes': [[0, 0, 100, 100], [10, 20, 300, 220], [100, 80, 900, 400], [0, 0, 5, 5]]}
        result = detector.normalize_result(output, 1000, 500, (.5, .5))
        self.assertEqual([d['label'] for d in result], ['fire', 'smoke'])
        self.assertEqual(result[0]['box'], [200, 160, 1000, 500])
        self.assertEqual(result[1]['box'], [20, 40, 600, 440])
        self.assertEqual(detector.normalize_result({'class_ids': [], 'scores': [], 'boxes': []}, 100, 100), [])

    def test_malformed_predictions_cannot_be_treated_as_successful_negative(self):
        for output in [
            {'class_ids': ['fire'], 'scores': [], 'boxes': []},
            {'class_ids': ['mask'], 'scores': [.9], 'boxes': [[1, 2, 3, 4]]},
            {'class_ids': ['fire'], 'scores': [float('nan')], 'boxes': [[1, 2, 3, 4]]},
            {'class_ids': ['fire'], 'scores': [.9], 'boxes': [[1, float('nan'), 3, 4]]},
        ]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                detector.normalize_result(output, 100, 100)

    def test_network_is_blocked_during_inference_and_restored_afterwards(self):
        original = socket.socket.connect
        with detector.offline_only():
            with self.assertRaisesRegex(RuntimeError, '本地资源'):
                socket.socket.connect(None, ('127.0.0.1', 8080))
        self.assertIs(socket.socket.connect, original)

    def test_validation_marker_is_invalidated_when_model_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hashes = {}
            for name in detector.MODEL_FILES:
                (root/name).write_bytes(b'test-model')
                hashes[name] = detector.digest(root/name)
            (root/'source.json').write_text(json.dumps({'sha256': hashes}))
            settings = {'enabled': True, 'model_dir': str(root), 'python': sys.executable,
                        'validation_file': str(root/'validation.json')}
            self.assertFalse(detector.readiness(settings)['ready'])
            detector.check_model(root)
            (root/'validation.json').write_text(json.dumps({'status': 'passed',
                                                          'fingerprint': detector.fingerprint(settings)}))
            self.assertTrue(detector.readiness(settings)['ready'])
            (root/detector.WEIGHT).write_bytes(b'corrupted-weight')
            self.assertFalse(detector.readiness(settings)['ready'])
            with self.assertRaisesRegex(ValueError, '校验失败'):
                detector.check_model(root)


class IndependentVisionToolsTests(unittest.TestCase):
    person = [{'label': 'person', 'confidence': .95, 'box': [10, 20, 90, 180]}]
    fire = [{'label': 'fire', 'confidence': .95, 'box': [100, 10, 200, 150]},
            {'label': 'smoke', 'confidence': .8, 'box': [70, 0, 220, 130]}]

    def run_case(self, task='统计人数，并检查烟火风险', person_failure=None,
                 fire_failure=None, fire_items=None):
        items = self.fire if fire_items is None else fire_items
        args = argparse.Namespace(task=task, image=str(agent.ROOT/'tests/fixtures/blank.jpg'),
                                  observation='', npu_detect=True, detections=None, tool_error=False,
                                  confirm=None, confirm_after=False, llm=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.multiple(agent, DATA=root, DB=root/'db.sqlite3',
                                REPORTS=root/'reports', EVIDENCE=root/'evidence'), \
                 patch.dict(agent.CONFIG, {'fire_smoke': {'enabled': True}}), \
                 patch.object(agent, 'sha256', return_value='test-hash'), \
                 patch.object(agent, 'run_npu_detector', side_effect=person_failure,
                              return_value=(self.person, {'image_size': [640, 427], 'threshold': .3})) as people, \
                 patch.object(agent, 'run_fire_smoke_detector', side_effect=fire_failure,
                              return_value=(items, {'image_size': [640, 427], 'threshold': .6,
                                                    'model_sha256': 'test-fire-hash'})) as smoke, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                record = agent.run_case(args)
                saved = json.loads((root/'reports'/f"{record['case_id']}.json").read_text())
                page = (root/'reports'/f"{record['case_id']}.html").read_text()
                self.assertEqual(agent.with_task_result(saved)['task_answer'], record['task_answer'])
                return record, page, people.call_count, smoke.call_count

    def test_combined_detection_preserves_both_tools_and_report_evidence(self):
        record, page, people, smoke = self.run_case()
        self.assertEqual((people, smoke), (1, 1))
        self.assertEqual(record['person_statistics']['estimated_count'], 1)
        self.assertEqual(record['fire_smoke_statistics']['classes']['fire']['count'], 1)
        self.assertEqual(record['risk_level'], record['response_assessment']['label'])
        self.assertEqual(record['vision_capabilities'], ['person', 'fire', 'smoke'])
        self.assertNotIn('blocked_exit', record['vision_missing_capabilities'])
        self.assertLess(len(record['task_answer']), 230)
        self.assertEqual(record['task_answer'].count(record['response_assessment']['label']), 1)
        self.assertEqual(record['task_answer'].count('/100'), 1)
        self.assertIn('撤离', record['task_answer'])
        self.assertIn('119', record['task_answer'])
        self.assertIn('1 处明火候选', record['task_answer'])
        self.assertIn('1 处烟雾候选', record['task_answer'])
        self.assertIn('#dc4f32', page)
        self.assertIn('fire 0.95', page)
        self.assertNotIn('尚未接入烟雾', page)

    def test_old_verbose_answer_is_compacted_without_rewriting_evidence_or_plan(self):
        import copy
        record, *_ = self.run_case()
        record.pop('answer_presentation_version')
        record.update(task_answer='旧版重复冗长的回答', evidence_answer='旧证据回答',
                      vision_missing_capabilities=['blocked_exit'])
        original = copy.deepcopy(record)
        result = agent.with_task_result(record)
        self.assertEqual(result['previous_task_answer'], original['task_answer'])
        self.assertEqual(result['previous_evidence_answer'], original['evidence_answer'])
        self.assertNotIn('blocked_exit', result['vision_missing_capabilities'])
        for key in ['request_plan', 'detections', 'response_assessment', 'vision_tools']:
            self.assertEqual(result[key], original[key])
        self.assertIn('119', result['task_answer'])
        self.assertEqual(agent.with_task_result(result), result)
        self.assertEqual(record, original)

    def test_smoke_failure_preserves_people_count_and_is_not_a_negative(self):
        record, page, *_ = self.run_case(fire_failure=RuntimeError('missing torch'))
        self.assertTrue(record['tool_error'])
        self.assertEqual(record['person_statistics']['estimated_count'], 1)
        self.assertIsNone(record['fire_smoke_statistics']['classes']['fire']['count'])
        self.assertIn('无法判断图片中的烟雾或明火', record['task_answer'])
        self.assertIn('missing torch', page)

    def test_person_failure_does_not_erase_fire_evidence(self):
        record, _, *_ = self.run_case(person_failure=RuntimeError('NPU unavailable'))
        self.assertIsNone(record['person_statistics']['estimated_count'])
        self.assertEqual(record['fire_smoke_statistics']['classes']['fire']['count'], 1)
        self.assertEqual(record['risk_level'], record['response_assessment']['label'])
        self.assertIn('1 处明火候选', record['task_answer'])

    def test_pure_fire_question_does_not_require_npu_or_count_people(self):
        record, _, people, smoke = self.run_case(task='这张图片的烟火情况怎么样？')
        self.assertEqual((people, smoke), (0, 1))
        self.assertEqual(record['task_kind'], 'risk_assessment')
        self.assertNotIn('人员目标', record['task_answer'])
        self.assertFalse(record['tool_error'])

    def test_scene_risk_and_advice_executes_fire_tool_and_preserves_history(self):
        record, _, people, smoke = self.run_case(task='这个情况危险吗,应该怎么办')
        self.assertEqual((people, smoke), (0, 1))
        self.assertTrue(record['request_plan']['asks_advice'])
        self.assertIn('1 处明火候选', record['task_answer'])
        # The original misrouted submission must not appear to have run a tool.
        old = dict(record, request_plan={'kind': 'general_question', 'show_people': False,
                                        'show_risk': False, 'use_detection': False},
                   task_answer='原来的文字回答', task_kind='general_question')
        enriched = agent.with_task_result(old)
        self.assertEqual(enriched['request_plan']['kind'], 'general_question')
        self.assertEqual(enriched['task_answer'], '原来的文字回答')

    def test_compound_scene_questions_and_conceptual_how_to_are_distinct(self):
        for task in ['这个情况危险吗，应该如何处理', '图片中着火了应该怎么办',
                     '这张图安全吗？下一步怎么应对', '这个情况应该如何处理']:
            with self.subTest(task=task):
                plan = agent.request_plan(task, has_image=True)
                self.assertEqual(plan['kind'], 'risk_assessment')
                self.assertTrue(plan['use_detection'])
                self.assertTrue(plan['asks_advice'])
        for task in ['有几个人，应该怎么疏散', '这张图有几个人应该怎么疏散',
                     '有几个人应该怎么疏散']:
            with self.subTest(task=task):
                plan = agent.request_plan(task, has_image=True)
                self.assertEqual(plan['kind'], 'person_count')
                self.assertFalse(plan['show_risk'])
        for task in ['如何判断火灾', '火灾是什么，有哪些危险', '发生火灾应该怎么办',
                     '如何判断图片中有没有烟雾', '怎么统计人数？', '如何准备应急包']:
            with self.subTest(task=task):
                plan = agent.request_plan(task, has_image=True)
                self.assertEqual(plan['kind'], 'general_question')
                self.assertFalse(plan['use_detection'])

    def test_attached_scene_risk_questions_execute_the_fire_tool(self):
        for task in ['这是什么风险', '这种情况怎么办', '这张图的风险是什么',
                     '这是什么危险', '这种情况应该怎么处理']:
            with self.subTest(task=task):
                record, _, people, smoke = self.run_case(task=task)
                self.assertEqual((people, smoke), (0, 1))
                self.assertEqual(record['task_kind'], 'risk_assessment')
                self.assertEqual(record['vision_tools']['fire_smoke']['status'], 'ok')
                self.assertIn('1 处明火候选', record['task_answer'])
        for task in ['这是什么风险', '这种情况怎么办']:
            plan = agent.request_plan(task, has_image=False)
            self.assertEqual(plan['kind'], 'general_question')
            self.assertFalse(plan['use_detection'])
        for task in ['解释这种风险的原理', '这种火灾应该如何预防',
                     '这张图检测烟火的原理是什么', '这种风险的评估方法是什么']:
            with self.subTest(task=task):
                plan = agent.request_plan(task, has_image=True)
                self.assertEqual(plan['kind'], 'general_question')
                self.assertFalse(plan['use_detection'])

    def test_success_with_zero_boxes_is_distinct_from_failure(self):
        record, _, *_ = self.run_case(task='检查是否有明火', fire_items=[])
        self.assertEqual(record['fire_smoke_statistics']['classes']['fire']['count'], 0)
        self.assertEqual(record['fire_smoke_statistics']['status'], 'available')
        self.assertIn('未检出达到检测阈值', record['task_answer'])
        self.assertIn('不能据此证明现场安全', record['task_answer'])


class ValidationGateTests(unittest.TestCase):
    def test_ready_requires_positive_categories_and_clean_negative_sample(self):
        import subprocess
        for mode, should_pass in [('empty', False), ('false_positive', False), ('valid', True)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                settings = dict(agent.CONFIG['fire_smoke'], validation_file=str(Path(tmp)/'validation.json'))
                case_root = Path(tmp)/'samples/fire-smoke'
                case_root.mkdir(parents=True)
                (case_root/'validation_cases.json').write_text(json.dumps([
                    {'image': 'positive1.jpg', 'positive': True},
                    {'image': 'positive2.jpg', 'positive': True},
                    {'image': 'positive3.jpg', 'positive': True},
                    {'image': 'test_00001095.jpg', 'positive': False}]))
                def predictions(command, **kwargs):
                    negative = 'test_00001095' in command[2]
                    labels = [] if mode == 'empty' or (negative and mode == 'valid') else ['fire', 'smoke']
                    items = [{'label': label, 'confidence': .95, 'box': [1, 2, 30, 40]} for label in labels]
                    payload = {'detections': items, 'image_size': [640, 480], 'elapsed_seconds': 1,
                               'model_sha256': 'test-fire-hash'}
                    return subprocess.CompletedProcess(command, 0, json.dumps(payload), '')
                with patch.object(setup, 'ROOT', Path(tmp)), patch.object(setup, 'check_model', return_value={'sha256': {detector.WEIGHT: 'test-fire-hash'}}), \
                     patch.object(setup, 'fingerprint', return_value={'test': 'fingerprint'}), \
                     patch.object(setup.subprocess, 'run', side_effect=predictions), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if should_pass:
                        setup.validate(settings)
                        marker = json.loads(Path(settings['validation_file']).read_text())
                        self.assertEqual(marker['status'], 'passed')
                        self.assertEqual(len(marker['samples']), 4)
                    else:
                        with self.assertRaises(RuntimeError):
                            setup.validate(settings)
                        self.assertFalse(Path(settings['validation_file']).exists())


if __name__ == '__main__':
    unittest.main()
