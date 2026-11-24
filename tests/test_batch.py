import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import agent
import webapp
from tools import batch_fire_smoke as batch
from tools import finalize_batch, init_knowledge


class DatasetEvaluationTests(unittest.TestCase):
    def test_duplicate_detection_is_false_positive_and_missed_object_is_false_negative(self):
        gt = [{'label': 'fire', 'box': [0, 0, 100, 100], 'difficult': False},
              {'label': 'smoke', 'box': [200, 200, 300, 300], 'difficult': False}]
        pred = [{'label': 'fire', 'box': [0, 0, 100, 100], 'confidence': .9},
                {'label': 'fire', 'box': [0, 0, 100, 100], 'confidence': .8}]
        metrics = batch.evaluate(pred, gt)
        self.assertEqual((metrics['fire']['tp'], metrics['fire']['fp']), (1, 1))
        self.assertEqual(metrics['smoke']['fn'], 1)

    def test_background_and_difficult_objects_are_handled_separately(self):
        pred = [{'label': 'smoke', 'box': [0, 0, 100, 100], 'confidence': .9}]
        gt = [{'label': 'smoke', 'box': [0, 0, 100, 100], 'difficult': True}]
        self.assertEqual(batch.evaluate(pred, gt)['smoke']['ignored_difficult'], 1)
        self.assertEqual(batch.evaluate(pred, [])['smoke']['fp'], 1)

    def test_image_presence_and_box_matching_measure_different_tasks(self):
        gt = [{'label': 'fire', 'box': [0, 0, 100, 100], 'difficult': False}]
        pred = [{'label': 'fire', 'box': [200, 200, 300, 300], 'confidence': .9}]
        boxes = batch.evaluate(pred, gt)
        self.assertEqual(boxes['fire']['tp'], 0)
        entry = {'evaluation': boxes, 'detections': pred, 'split': 'val'}
        images = finalize_batch.image_presence_metrics([entry])
        self.assertEqual(images['val']['fire']['tp'], 1)
        self.assertEqual(images['val']['fire']['recall'], 1)
        self.assertEqual(images['val']['smoke']['tn'], 1)

    def test_val_aggregate_does_not_include_training_images(self):
        metrics = batch.empty_metrics()
        for split in ['train', 'val', 'unlabeled']:
            entry = {'split': split}
            if split != 'unlabeled':
                entry['evaluation'] = {'fire': {'tp': 1, 'fp': 0, 'fn': 0, 'ignored_difficult': 0}}
            batch.update_metrics(metrics, entry)
        self.assertEqual(metrics['all_labeled']['fire']['tp'], 2)
        self.assertEqual(metrics['val']['fire']['tp'], 1)
        self.assertIsNone(batch.metrics_report(metrics)['val']['smoke']['recall'])

    def test_batch_api_paginates_filters_and_blocks_repeated_runs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(webapp, 'BATCHES', Path(tmp)):
            folder = Path(tmp)/'20261030-120000';folder.mkdir()
            (folder/'status.json').write_text(json.dumps({'id': folder.name, 'status': 'running'}))
            items = [{'index': i, 'name': f'{i}.jpg', 'score': i*30, 'level': i,
                      'evaluation': {'fire': {'fn': 1 if i == 0 else 0}}} for i in range(4)]
            (folder/'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items)+'{"incomplete"')
            result = webapp.batch_details(folder.name, {'offset': ['1'], 'limit': ['2']})
            self.assertEqual([x['index'] for x in result['items']], [1, 2])
            self.assertEqual(result['matched'], 4)
            self.assertEqual(webapp.batch_details(folder.name, {'level': ['missed']})['matched'], 1)
            self.assertEqual(webapp.batch_details(folder.name, {'level': ['3']})['items'][0]['score'], 90)
            with self.assertRaises(ValueError):webapp.submit_batch()
            with self.assertRaises(ValueError):webapp.batch_folder('../../outside')

    def test_web_worker_transmits_field_confirmations(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(webapp, 'JOBS', Path(tmp)):
            job_id = 'b'*32
            webapp.write_json(Path(tmp)/(job_id+'.json'), {'id': job_id, 'status': 'queued'})
            context = {'fire_type': 'ordinary', 'trained': True}
            def fake_run(command, **kwargs):
                value = json.loads(command[command.index('--safety-context')+1])
                self.assertEqual(value, context)
                Path(command[command.index('--result-file')+1]).write_text(json.dumps({'case_id': 'test'}))
                return argparse.Namespace(returncode=0)
            with patch.object(webapp.subprocess, 'run', side_effect=fake_run):
                webapp.worker(job_id, {'task': '检查火情', 'llm': False, 'safety_context': context}, None)
            self.assertEqual(webapp.read_json(Path(tmp)/(job_id+'.json'))['status'], 'done')

    def test_complete_batch_exports_all_reports_without_changing_detection_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.multiple(agent, DATA=root, DB=root/'db.sqlite3', REPORTS=root/'reports', EVIDENCE=root/'evidence'), contextlib.redirect_stdout(io.StringIO()):
                init_knowledge.main()
                path = root/'external.json'
                path.write_text(json.dumps([{'label': 'fire', 'confidence': .8, 'box': [1, 2, 10, 20]}]))
                args = argparse.Namespace(task='检查图片火情风险', image=None, observation='', npu_detect=False,
                                          detections=str(path), tool_error=False, confirm=None, confirm_after=False, llm=False)
                record = agent.run_case(args)
                folder = root/'batches'/'20261030-120000';folder.mkdir(parents=True)
                status = {'id': folder.name, 'status': 'done', 'total': 1, 'completed': 1, 'errors': 0,
                          'dataset_source': 'local-fixture', 'scope': 'fixture', 'metrics': batch.metrics_report(batch.empty_metrics())}
                (folder/'status.json').write_text(json.dumps(status))
                entry = {'index': 0, 'name': '<script>.jpg', 'case_id': record['case_id'], 'archive': 'fixture.zip',
                         'score': record['response_assessment']['score'], 'level': record['response_assessment']['level'],
                         'label': record['response_assessment']['label'], 'split': 'unlabeled', 'detections': record['detections']}
                (folder/'results.jsonl').write_text(json.dumps(entry)+'\n')
                finalize_batch.finalize(folder.name)
                saved = json.loads((agent.REPORTS/(record['case_id']+'.json')).read_text())
                self.assertEqual(saved['detections'], record['detections'])
                self.assertEqual(saved['response_assessment'], record['response_assessment'])
                self.assertEqual(saved['dataset_sample']['name'], '<script>.jpg')
                self.assertTrue(saved['knowledge_refs'])
                self.assertIn('fire_confidence', (folder/'results.csv').read_text(encoding='utf-8-sig'))
                self.assertIn('&lt;script&gt;.jpg', (folder/'summary.html').read_text())
                self.assertNotIn('<script>.jpg', (folder/'summary.html').read_text())
                detail = (agent.REPORTS/(record['case_id']+'.md')).read_text()
                self.assertIn('油锅起火', detail)
                self.assertIn('未确认/不满足', detail)


class OfflineKnowledgeTests(unittest.TestCase):
    def test_import_is_idempotent_and_retrieval_returns_substantive_cited_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.multiple(agent, DATA=root, DB=root/'db.sqlite3'), contextlib.redirect_stdout(io.StringIO()):
                init_knowledge.main();init_knowledge.main()
                db = agent.connect()
                self.assertEqual(db.execute('SELECT count(*) FROM docs').fetchone()[0], 6)
                refs = agent.retrieve(db, '油锅起火能用水吗')
                self.assertIn('不要泼水', refs[0]['excerpt'])
                self.assertEqual(refs[0]['review_status'], 'pending_local_review')
                line = Path(refs[0]['source']).read_text().splitlines()[refs[0]['line']-1]
                self.assertEqual(line, refs[0]['excerpt'])
                manifest = json.loads((root/'knowledge_manifest.json').read_text())
                self.assertFalse(any(d['official_reference_downloaded'] for d in manifest['documents']))
                db.close()


if __name__ == '__main__':
    unittest.main()
