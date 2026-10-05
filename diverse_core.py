"""Export the diverse documentary core, leaving current web journalism separate."""

import argparse
import collections
import gzip
import json
from pathlib import Path

import datawork as d
import diverse
import newswork as nw

ROOT = Path(__file__).resolve().parent
GENRES = {'reference', 'fiction', 'nonfiction', 'parliament'}
SPLITS = ('train', 'validation', 'test')


def load(source):
    manifest = diverse.strict.read(source / 'manifest.json')
    verification = diverse.strict.read(source / 'verification.json')
    if not verification['verified'] or verification['manifest_sha256'] != nw.digest_file(source / 'manifest.json'):
        raise ValueError('Source mixture needs a matching verification report')
    for name in [s + '.jsonl.gz' for s in SPLITS] + ['parent_documents.jsonl.gz']:
        if nw.digest_file(source / name) != manifest['outputs'][name]:
            raise ValueError('Source export changed')
    rows = [r for split in SPLITS for r in diverse.strict.rows(source / (split + '.jsonl.gz')) if r['genre'] in GENRES]
    ids = {r['parent_document_id'] for r in rows}
    parents = [r for r in diverse.strict.rows(source / 'parent_documents.jsonl.gz') if r['id'] in ids]
    return rows, parents


def build(source, output):
    if output == source or source in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError('Use a fresh separate output directory')
    rows, parents = load(source)
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        diverse.write_rows(output / (split + '.jsonl.gz'), (r for r in rows if r['split'] == split))
    diverse.write_rows(output / 'parent_documents.jsonl.gz', parents)
    nw.save_json(output / 'licenses.snapshot.json', diverse.strict.read(ROOT / 'config/licenses.json'))
    nw.save_json(output / 'diversity.json', diverse.summarize(rows))
    (output / 'ATTRIBUTION.txt').write_text((source / 'ATTRIBUTION.txt').read_text())
    manifest = {'processing_complete': True, 'source': str(source), 'input_manifest_sha256': nw.digest_file(source / 'manifest.json'),
                'input_verification_sha256': nw.digest_file(source / 'verification.json'), 'script_sha256': nw.digest_file(Path(__file__)),
                'included_genres': sorted(GENRES), 'passages': len(rows), 'words': sum(r['whitespace_words'] for r in rows), 'parent_documents': len(parents),
                'confidence': 'Diverse human-origin likelihood from identified historical authors, a documented 2016 reference corpus and named historical parliamentary prose; no measured purity guarantee',
                'limitations': ['Web journalism remains in the broader mixture, excluded from this documentary core.', 'Wikipedia article revision IDs and individual authorship are unverified; release provenance is the confidence basis.', 'Gutenberg ebook bodies derive from historical works; present-day digitization versions differ from original publication years.', 'Source licensing uncertainty and historical/benchmark formatting remain recorded.', 'Exact overlapping subset; do not append to the broad mixture or older core as additional unique data.'],
                'summary': diverse.summarize(rows), 'outputs': {p.name: nw.digest_file(p) for p in output.iterdir() if p.is_file()}}
    nw.save_json(output / 'manifest.json', manifest)
    print(json.dumps({'passages':len(rows), 'words':manifest['words'], 'genres':manifest['summary']['by_genre']},indent=2),flush=True)


def verify(output):
    manifest = diverse.strict.read(output / 'manifest.json')
    source = Path(manifest['source'])
    if nw.digest_file(source / 'manifest.json') != manifest['input_manifest_sha256'] or nw.digest_file(source / 'verification.json') != manifest['input_verification_sha256']:
        raise ValueError('Source provenance changed')
    for name, expected in manifest['outputs'].items():
        if Path(name).name != name or nw.digest_file(output / name) != expected:
            raise ValueError('Output hash mismatch')
    expected, parents = load(source)
    actual = [r for split in SPLITS for r in diverse.strict.rows(output / (split + '.jsonl.gz'))]
    if {r['id']:r for r in actual} != {r['id']:r for r in expected} or len(actual) != len(expected):
        raise ValueError('Not an exact unchanged source subset')
    if list(diverse.strict.rows(output / 'parent_documents.jsonl.gz')) != parents:
        raise ValueError('Parent export differs')
    groups, texts = {}, set()
    for split in SPLITS:
        for r in diverse.strict.rows(output / (split + '.jsonl.gz')):
            if r['split'] != split or r['genre'] not in GENRES or r['dataset'] not in {'hansard','wikitext2_raw','gutenberg_selected'}:
                raise ValueError('Genre/dataset/split scope differs')
            h = d.text_hash(d.normalized(r['text']))
            if h in texts or groups.setdefault(r['split_group'], split) != split:
                raise ValueError('Duplicate text or group leakage')
            texts.add(h)
    if diverse.summarize(actual) != manifest['summary'] or sum(r['whitespace_words'] for r in actual) != manifest['words']:
        raise ValueError('Count or diversity mismatch')
    nw.save_json(output / 'verification.json', {'verified':True,'manifest_sha256':nw.digest_file(output/'manifest.json'),'verifier_sha256':nw.digest_file(Path(__file__)), 'passages':len(actual),'words':manifest['words'],'checks':['Pinned verified parent mixture','Exact unchanged subset','Parent metadata','Unique text and source-group partitions','Genre and word counts'],'scope':'Integrity and documentary source policy; no measured human-only guarantee'})
    print('Verified',len(actual),'passages',manifest['words'],'words',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--source',type=Path,default=ROOT/'data/corpus-human-diverse-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'data/corpus-human-diverse-core-v1')
    args=parser.parse_args()
    if args.command=='build':build(args.source.resolve(),args.output.resolve())
    else:verify(args.output.resolve())
