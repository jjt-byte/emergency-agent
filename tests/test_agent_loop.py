"""Agent integration: evidence, bounded tool selection and memory isolation."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent
import agent_loop
import webapp


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        model_digest = patch.object(agent, 'model_digest', return_value='software-test-model-digest')
        model_digest.start()
        self.addCleanup(model_digest.stop)

    person = [{'label': 'person', 'confidence': .95, 'box': [10, 20, 90, 180]}]
    fire = [{'label': 'fire', 'confidence': .95, 'box': [0, 0, 600, 400]}]

    def run_agent(self, decisions, task='有几个人，有火吗，应该怎么办？', context=None,
                  fire_failure=None, image=True, progress=False):
        args = argparse.Namespace(task=task, image=str(agent.ROOT/'tests/fixtures/blank.jpg') if image else None,
                                  observation='', npu_detect=image, detections=None, tool_error=False,
                                  confirm=None, confirm_after=False, llm=True, agent=True,
                                  safety_context={}, agent_context=context or {})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress_path = root/'agent.progress'
            if progress:
                args.progress_file = str(progress_path)
            with patch.multiple(agent, DATA=root, DB=root/'db.sqlite3', REPORTS=root/'reports', EVIDENCE=root/'evidence'), \
                 patch.object(agent_loop, 'complete', side_effect=decisions) as llm, \
                 patch.object(agent, 'run_npu_detector', return_value=(self.person, {'image_size': [640, 427]})) as people, \
                 patch.object(agent, 'run_fire_smoke_detector', side_effect=fire_failure,
                              return_value=(self.fire, {'image_size': [640, 427]})) as fire, \
                 contextlib.redirect_stdout(io.StringIO()):
                record = agent.run_case(args)
                saved = json.loads((root/'reports'/f"{record['case_id']}.json").read_text())
                self.assertEqual(saved['task_answer'], record['task_answer'])
                self.assertEqual(agent.with_task_result(saved)['task_answer'], record['task_answer'])
                states = [call.args[1] for call in llm.call_args_list]
                result = (record, people.call_count, fire.call_count, states)
                if progress:
                    return (*result, json.loads(progress_path.read_text()))
                return result

    def test_model_observes_evidence_then_selects_another_tool_and_asks(self):
        record, people, fire, states = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'detect_fire_smoke'}]},
            {'action': 'tools', 'tools': [{'name': 'detect_people'}, {'name': 'search_knowledge', 'query': '火灾撤离'}]},
            {'action': 'ask', 'answer': '图片无法确认是否有人被困。', 'question': '现场是否有人无法自行撤离？'}])
        self.assertEqual((people, fire), (1, 1))
        self.assertEqual(states[1]['tools']['fire_smoke']['status'], 'ok')
        self.assertEqual(states[2]['tools']['person']['status'], 'ok')
        self.assertIn('119', record['task_answer'])
        self.assertIn('现场是否有人无法自行撤离', record['task_answer'])
        self.assertEqual(record['agent_execution']['status'], 'completed')
        self.assertTrue(any(x['title'] == '本地知识检索' and x['source'] == 'qwen' for x in record['agent_trace']))

    def test_live_progress_includes_tool_choice_without_answer_draft(self):
        record, *_rest, progress = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'detect_people'}]},
            {'action': 'finish', 'answer': '估计有1人。'}],
            task='这张图有几个人？', progress=True)
        self.assertEqual(progress['decisions'][0], {
            'action': 'tools', 'tools': [{'name': 'detect_people'}]})
        self.assertEqual(progress['decisions'][-1]['action'], 'finish')
        self.assertNotIn('answer', json.dumps(progress))
        self.assertTrue(any(step['title'] == '第 1 轮工具决策' and step['source'] == 'qwen'
                            for step in progress['steps']))

    def test_model_cannot_invent_count_or_lower_the_emergency_alert(self):
        record, *_ = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'detect_people'}, {'name': 'detect_fire_smoke'}]},
            {'action': 'tools', 'tools': [{'name': 'search_knowledge', 'query': '撤离报警'}]},
            {'action': 'finish', 'answer': '这里有6个人，现场完全安全。'}])
        self.assertIn('1 人', record['task_answer'])
        self.assertNotIn('6个人', record['task_answer'])
        self.assertIn('紧急优先级预警', record['task_answer'])
        self.assertIn('119', record['task_answer'])
        self.assertEqual(record['agent_execution']['status'], 'fallback')

    def test_detector_failure_stays_unknown_and_model_cannot_claim_no_fire(self):
        record, *_ = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'detect_fire_smoke'}]},
            {'action': 'finish', 'answer': '没有火，放心。'}],
            task='这张图有火吗？', fire_failure=RuntimeError('test failure'))
        self.assertIsNone(record['response_assessment']['score'])
        self.assertIn('无法判断', record['task_answer'])
        self.assertNotIn('放心', record['task_answer'])
        self.assertEqual(record['agent_execution']['status'], 'fallback')

    def test_repeated_tool_calls_end_and_detector_runs_once(self):
        record, people, fire, states = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'detect_people'}]},
            {'action': 'tools', 'tools': [{'name': 'detect_people'}]},
            {'action': 'finish', 'answer': '估计有1人，可能存在漏检。'}], task='这张图有几个人？')
        self.assertEqual((people, fire), (1, 0))
        self.assertEqual(len(states), 3)
        self.assertTrue(any(x['title'] == '停止重复调用' for x in record['agent_trace']))

    def test_cache_reused_only_for_same_image_and_model_configuration(self):
        cache = {'state': {'status': 'ok', 'source': 'damoyolo_person_cpu',
                           'result': {'image_size': [640, 427], 'detections': self.person}},
                 'case_id': '20261030-000000-aaaaaa', 'fingerprint': agent_loop.fingerprint('detect_people')}
        ctx = {'image_sha256': agent.sha256(agent.ROOT/'tests/fixtures/blank.jpg'), 'cached_tools': {'person': cache},
               'history': [{'question': '上一问', 'answer': '已有结果'}]}
        decisions = [{'action': 'tools', 'tools': [{'name': 'detect_people'}]},
                     {'action': 'finish', 'answer': '约1人。'}]
        r, people, _, _ = self.run_agent(decisions, task='有几个人？', context=ctx)
        self.assertEqual(people, 0)
        self.assertEqual(r['vision_tools']['person']['reused_from_case_id'], cache['case_id'])
        ctx['image_sha256'] = 'different-image'
        r, people, _, _ = self.run_agent(decisions, task='有几个人？', context=ctx)
        self.assertEqual(people, 1)
        self.assertNotIn('reused_from_case_id', r['vision_tools']['person'])

    def test_ordinary_question_keeps_model_answer_and_does_not_detect(self):
        answer = '可以准备饮用水、手电筒和常用药品。'
        r, people, fire, _ = self.run_agent([{'action': 'finish', 'answer': answer}], task='应急包放什么？', image=False)
        self.assertEqual((people, fire), (0, 0))
        self.assertEqual(r['task_answer'], answer)

    def test_invalid_or_unbounded_commands_are_rejected(self):
        for decision in [{'action': 'tools', 'tools': [{'name': 'shell', 'query': 'rm'}]},
                         {'action': 'tools', 'tools': []}, {'action': 'ask', 'question': ''},
                         {'action': 'tools', 'tools': [{'name': 'detect_people', 'image': '/etc/passwd'}]}]:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                agent_loop.validate_decision(decision)

    def test_refusing_to_use_an_extinguisher_is_not_mistaken_for_permission(self):
        r, *_ = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'detect_fire_smoke'}]},
            {'action': 'finish', 'answer': '您不应自行使用灭火器。请撤离，在安全位置拨打119。'}], task='图片中有火吗？')
        self.assertEqual(r['agent_execution']['status'], 'completed')
        self.assertIn('不应自行使用灭火器', r['task_answer'])
        self.assertIsNotNone(agent_loop.check_answer('可以使用灭火器扑救。', r))

    def test_same_scene_safety_memory_does_not_turn_ordinary_qa_into_counts(self):
        cache = {'state': {'status': 'ok', 'source': 'damoyolo_cpu',
                           'result': {'image_size': [640, 427], 'detections': self.fire}},
                 'case_id': '20261030-000000-aaaaaa', 'fingerprint': agent_loop.fingerprint('detect_fire_smoke')}
        ctx = {'image_sha256': agent.sha256(agent.ROOT/'tests/fixtures/blank.jpg'),
               'cached_tools': {'fire_smoke': cache}, 'scene_reused': True}
        answer = '离线是指不依赖互联网连接。'
        r, people, fire, _ = self.run_agent([{'action': 'finish', 'answer': answer}],
                                          task='解释一下什么是离线运行？', context=ctx)
        self.assertEqual((people, fire), (0, 0))
        self.assertEqual(r['task_answer'], answer)
        self.assertFalse(r['request_plan']['show_people'])
        self.assertFalse(r['request_plan']['show_risk'])

    def test_extinguisher_followup_reuses_scene_and_retrieves_guidance(self):
        cache = {'state': {'status': 'ok', 'source': 'damoyolo_cpu',
                           'result': {'image_size': [640, 427], 'detections': self.fire}},
                 'case_id': '20261030-000000-aaaaaa', 'fingerprint': agent_loop.fingerprint('detect_fire_smoke')}
        ctx = {'image_sha256': agent.sha256(agent.ROOT/'tests/fixtures/blank.jpg'),
               'cached_tools': {'fire_smoke': cache}, 'scene_reused': True}
        r, people, fire, _ = self.run_agent([
            {'action': 'tools', 'tools': [{'name': 'search_knowledge', 'query': '灭火器使用条件'}]},
            {'action': 'finish', 'answer': '请立即撤离，在安全位置拨打119，不应自行使用灭火器。'}],
            task='那我能自己用灭火器吗？', context=ctx)
        self.assertEqual((people, fire), (0, 0))
        self.assertFalse(r['request_plan']['show_people'])
        self.assertEqual(r['risk_level'], '紧急优先级预警')
        self.assertTrue(any(x['title'] == '本地知识检索' and x['source'] == 'qwen' for x in r['agent_trace']))


class ConversationTests(unittest.TestCase):
    def test_session_paths_and_context_are_server_owned_and_scene_can_be_cleared(self):
        for bad in ['../../config', 'x'*32, None]:
            with self.assertRaises(ValueError):
                webapp.conversation_path(bad)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence, reports = root/'evidence', root/'reports'
            evidence.mkdir(); reports.mkdir()
            image = evidence/'scene.jpg'
            image.write_bytes(b'first image')
            case = '20261030-000000-aaaaaa'
            record = {'case_id': case, 'task': '原问题', 'task_answer': '原回答',
                      'image': {'path': str(image), 'sha256': agent.sha256(image)}}
            (reports/(case+'.json')).write_text(json.dumps(record))
            session = {'id': 'a'*32, 'case_ids': [case], 'scene_case_id': case}
            with patch.multiple(agent, REPORTS=reports, EVIDENCE=evidence):
                context, inherited = webapp.conversation_context(session)
                self.assertEqual(inherited, image)
                self.assertTrue(context['scene_reused'])
                self.assertEqual(context['history'][0]['answer'], '原回答')
                context, inherited = webapp.conversation_context(session, clear_scene=True)
                self.assertIsNone(inherited)
                self.assertNotIn('image_sha256', context)
                new_image = evidence/'new.jpg'; new_image.write_bytes(b'new image')
                context, inherited = webapp.conversation_context(session, image=new_image)
                self.assertEqual(inherited, new_image)
                self.assertFalse(context.get('scene_reused', False))
                self.assertEqual(context['cached_tools'], {})


if __name__ == '__main__':
    unittest.main()
