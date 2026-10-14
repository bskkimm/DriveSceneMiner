"""Compare saved providers on identical shared evidence and explicitly reviewed time."""
import json
import math
from pathlib import Path

from jevsceneminer.inference.runs import read_rows, script_hash
from .evaluate import spans


def _union(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def _reviewed(gt, candidate, domain, key):
    def checked(doc):
        values = spans(doc, key)
        previous = None
        for a,b,_ in values:
            if not math.isfinite(a) or not math.isfinite(b) or a >= b:
                raise ValueError('comparison requires valid scene intervals')
            if previous is not None and a < previous:
                raise ValueError('comparison rejects overlapping or unordered scene intervals')
            previous = b
        return values
    reviewed = []
    for a, b, label in checked(gt):
        if label in (None, '', 'unknown', 'unlabeled', 'unlabelled'):
            continue
        for lo, hi in domain:
            if min(b, hi) > max(a, lo):
                reviewed.append((max(a, lo), min(b, hi), label))
    duration = sum(b-a for a,b,_ in reviewed)
    covered = correct = 0.
    predictions = checked(candidate)
    for a,b,label in reviewed:
        for pa,pb,predicted in predictions:
            overlap = max(0., min(b,pb)-max(a,pa))
            if predicted not in (None, '', 'unknown', 'unlabeled', 'unlabelled'):
                covered += overlap
                if predicted == label:
                    correct += overlap
    return dict(reviewed_seconds=duration, predicted_reviewed_seconds=covered,
                unknown_seconds=max(0., duration-covered), correct_seconds=correct,
                agreement=correct/duration if duration else None)


def compare_runs(runs, out, gt_dir=None):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('comparison output already exists; choose a new directory')
    if len(runs) < 2:
        raise ValueError('compare requires at least two named run folders')
    sessions = None
    data = {}
    for name, root in runs.items():
        root = Path(root).resolve()
        if root == out or root in out.parents or out in root.parents:
            raise ValueError('comparison output must be separate from run folders')
        metas = {path.stem:json.loads(path.read_text()) for path in (root/'meta').glob('*.json')}
        if not metas or sessions is not None and set(metas) != sessions:
            raise ValueError('all compared runs must contain the same sessions')
        sessions = set(metas)
        data[name] = {}
        for sid, meta in metas.items():
            rows = read_rows(root/'steps'/f'{sid}.jsonl')
            scenes = json.loads((root/'scenes'/f'{sid}.json').read_text())
            runtime = root/'runtime'/f'{sid}.json'
            data[name][sid] = dict(meta=meta, rows=rows, scenes=scenes,
                                  runtime=json.loads(runtime.read_text()) if runtime.exists() else {})
    report = dict(schema_version='jsm-comparison-1', same_scripts=True,
                  question_definitions_verified=True, sessions={}, runs={})
    reviewed_by_run = {name:{key:dict(reviewed_seconds=0.,predicted_reviewed_seconds=0.,
                                    unknown_seconds=0.,correct_seconds=0.) for key in ('lateral','longitudinal')}
                       for name in runs}
    for sid in sorted(sessions):
        indexes = {name:{row['t_ns']:row for row in item[sid]['rows']} for name,item in data.items()}
        common = sorted(set.intersection(*(set(index) for index in indexes.values())))
        if not common:
            raise ValueError('runs have no common prepared timestamps')
        selected = [next(iter(indexes.values()))[timestamp] for timestamp in common]
        for index in indexes.values():
            if any(index[t]['script'] != selected[i]['script'] for i,t in enumerate(common)):
                raise ValueError('compared scripts differ at a common timestamp')
        questions = [item[sid]['scenes'].get('generator',{}).get('questions_sha256') for item in data.values()]
        if len({value for value in questions if value}) > 1:
            raise ValueError('compared question definitions differ')
        report['question_definitions_verified'] &= all(questions)
        first = next(iter(data.values()))[sid]['meta']
        step = first['step_s']
        if any(item[sid]['meta']['step_s'] != step for item in data.values()):
            raise ValueError('comparison requires identical inference cadence')
        domain = _union([(max(first['start_ns']/1e9,t/1e9-step/2),
                          min(first['end_ns']/1e9,t/1e9+step/2)) for t in common])
        report['sessions'][sid] = dict(common_samples=len(common), common_script_sha256=script_hash(selected),
                                       intervals_s=domain)
        gt_path = Path(gt_dir)/f'{sid}.json' if gt_dir else None
        if gt_path and gt_path.is_file():
            gt = json.loads(gt_path.read_text())
            for name, item in data.items():
                for key in ('lateral','longitudinal'):
                    metrics = _reviewed(gt,item[sid]['scenes'],domain,key)
                    for field in reviewed_by_run[name][key]:
                        reviewed_by_run[name][key][field] += metrics[field]
    for name, item in data.items():
        runtime = [value['runtime'] for value in item.values()]
        def total(field):
            values = [r.get(field) for r in runtime]
            return sum(values) if all(isinstance(v,(int,float)) for v in values) else None
        reviewed = reviewed_by_run[name]
        for metrics in reviewed.values():
            metrics['agreement'] = metrics['correct_seconds']/metrics['reviewed_seconds'] if metrics['reviewed_seconds'] else None
        report['runs'][name] = dict(providers=sorted({value['meta'].get('inference_provider','jev') for value in item.values()}),
            models=sorted({row.get('model','unknown') or 'unknown' for value in item.values() for row in value['rows']}),
            fresh_answers=total('fresh_answers'), inference_seconds=total('inference_seconds'),
            estimated_cost_usd=total('estimated_cost_usd'), reviewed=reviewed)
    report['notes'] = ['Agreement uses only explicitly labeled GT within common inference cells; unlabeled time is excluded.',
                       'Partial trials do not establish full-log precision/recall or maneuver boundary accuracy.',
                       'Wall time and cost include cached/fresh counts; different sample sets and cache hits limit speed comparisons.',
                       'Legacy runs without question hashes cannot establish identical question definitions.']
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Decision-provider comparison','', '| Run | Provider | Fresh answers | Inference (s) | Estimated cost (USD) | Reviewed lateral agreement |',
           '| --- | --- | --- | --- | --- | --- |']
    for name, item in report['runs'].items():
        value=item['reviewed']['lateral']['agreement']
        lines.append(f"| {name} | {', '.join(item['providers'])} | {item['fresh_answers']} | {item['inference_seconds']} | {item['estimated_cost_usd']} | {value if value is not None else 'unknown'} |")
    lines += ['', *report['notes'], '']
    (out/'report.md').write_text('\n'.join(lines))
    return report
