"""Decision boundaries and action eligibility, independent of model inference."""
import unittest
import copy
import agent
import triage


class TriageTests(unittest.TestCase):
    def record(self, detections=(), status='ok', size=(1000, 1000)):
        return {'detections': list(detections), 'vision_tools': {'fire_smoke':
                {'status': status, 'result': {'image_size': list(size)}}}}

    def small_fire(self):
        return {'label': 'fire', 'confidence': .61, 'box': [0, 0, 10, 10]}

    def test_no_detection_never_means_safe_and_failure_is_not_zero(self):
        result = triage.assess(self.record())
        self.assertEqual(result['score'], 0)
        self.assertIn('不能证明安全', result['notice'])
        self.assertFalse(result['extinguisher_eligible'])
        self.assertIsNone(triage.assess(self.record(status='failed'))['score'])
        self.assertIsNone(triage.assess(self.record(status='skipped'))['level'])

    def test_fire_alert_needs_every_field_condition_and_matching_material(self):
        context = {name: True for name in triage.FIRE_CONDITIONS}
        context['fire_type'] = 'ordinary'
        record = self.record([self.small_fire()])
        result = triage.assess(record, context)
        self.assertEqual(result['level'], 2)
        self.assertTrue(result['extinguisher_eligible'])
        for name in triage.FIRE_CONDITIONS:
            with self.subTest(condition=name):
                self.assertFalse(triage.assess(record, dict(context, **{name: False}))['extinguisher_eligible'])
        for material in ['unknown', 'battery', 'gas']:
            self.assertFalse(triage.assess(record, dict(context, fire_type=material))['extinguisher_eligible'])

    def test_emergency_and_smoke_only_do_not_authorize_extinguishing(self):
        context = dict.fromkeys(triage.FIRE_CONDITIONS, True)
        context['fire_type'] = 'ordinary'
        large_fire = {'label': 'fire', 'confidence': .99, 'box': [0, 0, 1000, 1000]}
        result = triage.assess(self.record([large_fire]), context)
        self.assertEqual(result['level'], 3)
        self.assertFalse(result['extinguisher_eligible'])
        self.assertIn('优先', result['actions'][0])
        smoke = {'label': 'smoke', 'confidence': .7, 'box': [0, 0, 10, 10]}
        self.assertFalse(triage.assess(self.record([smoke]), context)['extinguisher_eligible'])

    def test_duplicates_do_not_raise_score_and_scaled_boxes_have_same_score(self):
        fire = self.small_fire()
        one = triage.assess(self.record([fire]))
        many = triage.assess(self.record([fire]*100))
        self.assertEqual(one['score'], many['score'])
        scaled = dict(fire, box=[0, 0, 20, 20])
        self.assertEqual(one['score'], triage.assess(self.record([scaled], size=(2000, 2000)))['score'])
        self.assertEqual(0, triage.assess(self.record([{'label': 'person', 'confidence': .99}]))['score'])

    def test_context_validation_rejects_coerced_booleans_and_unknown_fields(self):
        for context in [None, {'trained': 'false'}, {'fire_type': 'water'}, {'auto_call': True}]:
            with self.subTest(context=context), self.assertRaises(ValueError):
                triage.validate_context(context)

    def test_main_risk_heading_matches_image_alert_and_keeps_review_separate(self):
        record = self.record([
            {'label': 'smoke', 'confidence': .743166, 'box': [3.35, 0, 949.96, 528.29]},
            {'label': 'fire', 'confidence': .729499, 'box': [75.44, 24.22, 888.39, 432.59]},
        ], size=(1002, 646))
        record.update(task='这是什么风险', image={'path': 'test.jpg'}, observation='',
                      risk_level='疑似风险待人工确认', rule_reason='旧版独立置信度阈值',
                      human_confirmation='未确认', answer_version=3,
                      task_answer='原来互相矛盾的风险回答',
                      request_plan=agent.request_plan('这是什么风险', True))
        record['response_assessment'] = triage.assess(record)
        original = copy.deepcopy(record)
        result = agent.with_task_result(record)
        self.assertEqual(result['response_assessment']['score'], 90)
        self.assertEqual(result['risk_level'], '紧急优先级预警')
        self.assertIn('优先撤离并报警', result['rule_reason'])
        self.assertIn('风险提示：紧急优先级预警', result['task_answer'])
        self.assertNotIn('疑似风险待人工确认', result['task_answer'])
        self.assertEqual(result['human_confirmation'], '未确认')
        self.assertEqual(result['previous_risk_summary']['task_answer'], original['task_answer'])
        self.assertEqual(result['detections'], original['detections'])
        self.assertEqual(result['response_assessment'], original['response_assessment'])
        self.assertEqual(record, original)
        self.assertEqual(agent.with_task_result(result), result)

    def test_low_alert_and_unavailable_detection_have_distinct_main_headings(self):
        for record, expected in [
            (self.record(), '未检出或线索不足'),
            (self.record([self.small_fire()]), '高优先级预警'),
            (self.record(status='failed'), '无法评估：检测未完成'),
            (self.record(status='skipped'), '无法评估：检测未完成'),
        ]:
            assessment = triage.assess(record)
            title, reason = triage.risk_summary(assessment)
            self.assertEqual(title, expected)
            self.assertTrue(reason)

    def test_zero_image_score_does_not_downgrade_reported_visible_fire(self):
        record = self.record()
        record.update(task='检查现场火情', image={'path': 'test.jpg'}, observation='现场有明火',
                      risk_level='高风险待人工确认', rule_reason='出现明火线索')
        record['response_assessment'] = triage.assess(record)
        result = agent.with_task_result(record)
        self.assertEqual(result['response_assessment']['score'], 0)
        self.assertEqual(result['risk_level'], '高优先级预警')
        self.assertIn('现场补充观察触发升级', result['rule_reason'])
        self.assertIn('现场补充观察已触发升级', result['task_answer'])
        self.assertIn('向安全区域撤离', result['task_answer'])
        self.assertIn('119', result['task_answer'])


if __name__ == '__main__':
    unittest.main()
