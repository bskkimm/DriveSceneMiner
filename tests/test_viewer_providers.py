import json

from jevsceneminer.viewer import Site


def test_viewer_keeps_each_candidates_probabilities_separate(tmp_path):
    roots=[]
    for name,label,provider in [('base','keep_lane','jev'),('candidate','turn_right','microsoft')]:
        root=tmp_path/name; roots.append(root)
        for folder in ('meta','steps','scenes'): (root/folder).mkdir(parents=True)
        (root/'meta'/'s.json').write_text(json.dumps({'inference_provider':provider,'inference_model':name}))
        (root/'steps'/'s.jsonl').write_text(json.dumps({'t_ns':1,'script':'same input','facts':{'large':'audit'},
            'lateral':label,'lateral_probs':{label:1.},'provider':provider})+'\n')
        (root/'scenes'/'s.json').write_text(json.dumps({'scenes':[]}))
    site=Site(roots[0],None,None,{'jev':roots[0]/'scenes','microsoft':roots[1]})
    data=site.session('s')
    assert data['run_steps']['jev'][0]['lateral']=='keep_lane'
    assert data['run_steps']['microsoft'][0]['lateral']=='turn_right'
    assert data['run_meta']['microsoft']['inference_provider']=='microsoft'
    assert 'script' not in data['run_steps']['microsoft'][0] and 'facts' not in data['run_steps']['microsoft'][0]


def test_detached_scene_document_does_not_borrow_base_raw_probabilities(tmp_path):
    for folder in ('meta','steps','scenes','detached'): (tmp_path/folder).mkdir()
    (tmp_path/'meta'/'s.json').write_text('{}')
    (tmp_path/'steps'/'s.jsonl').write_text(json.dumps({'t_ns':1,'script':'input','lateral':'keep_lane'})+'\n')
    (tmp_path/'detached'/'s.json').write_text(json.dumps({'scenes':[]}))
    data=Site(tmp_path,None,None,{'candidate':tmp_path/'detached'}).session('s')
    assert data['run_steps']['candidate']==[]
