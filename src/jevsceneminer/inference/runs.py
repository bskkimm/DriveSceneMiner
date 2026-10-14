"""Fork prepared evidence without editing the source or mixing provider answers."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ANSWER_FIELDS = {'lateral', 'longitudinal', 'lateral_probs', 'longitudinal_probs',
                 'provider', 'model', 'requested_model', 'input_tokens', 'output_tokens', 'cache_key'}


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    previous = None
    for row in rows:
        if not isinstance(row, dict) or type(row.get('t_ns')) is not int or not isinstance(row.get('script'), str):
            raise ValueError('prepared rows require integer t_ns and readable script')
        if previous is not None and row['t_ns'] <= previous:
            raise ValueError('prepared timestamps must strictly increase')
        previous = row['t_ns']
    return rows


def script_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row['t_ns'], row['script']], ensure_ascii=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def fork_run(source, destination, args, resolve_settings):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('output destination already exists; resume by classifying that folder directly')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('output must be separate from the prepared source folder')
    metas = sorted((source/'meta').glob('*.json'))
    if not metas:
        raise ValueError('source has no prepared session metadata')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+'.', dir=destination.parent) as scratch:
        temporary = Path(scratch)/'run'
        temporary.mkdir()
        for sub in ('meta', 'indicator', 'tags', 'camera', 'rules', 'cache'):
            if (source/sub).is_dir():
                shutil.copytree(source/sub, temporary/sub)
        (temporary/'steps').mkdir()
        for path in metas:
            meta = json.loads(path.read_text())
            rows = read_rows(source/'steps'/f'{path.stem}.jsonl')
            if getattr(args, 'from_s', None) is not None:
                lo = meta['start_ns'] + round(args.from_s*1e9)
                hi = meta['start_ns'] + round(args.to_s*1e9)
                rows = [row for row in rows if lo <= row['t_ns'] < hi]
            if not rows:
                raise ValueError('selected range contains no prepared samples')
            clean = [{key:value for key,value in row.items() if key not in ANSWER_FIELDS} for row in rows]
            (temporary/'steps'/f'{path.stem}.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in clean))
            settings = resolve_settings(args, meta)
            meta.update(inference_provider=settings.provider, inference_model=settings.model,
                        inference_endpoint=settings.endpoint, source_run=str(source),
                        prepared_script_sha256=script_hash(clean),
                        prepared_range_s=[getattr(args,'from_s',None), getattr(args,'to_s',None)])
            (temporary/'meta'/path.name).write_text(json.dumps(meta)+'\n')
        temporary.rename(destination)
    return destination
