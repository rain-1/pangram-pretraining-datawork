"""Experimental text-only translation baseline; never certifies corpus authorship."""

import argparse
import collections
import datetime as dt
import importlib.metadata
import json
import math
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import datawork as d
import newswork as nw

ROOT = Path(__file__).resolve().parent


def prepare(records, spans, minimum_words=400):
    # Existing sentence and span exports were partitioned separately. Regroup
    # the complete release before filtering or selecting binary representatives.
    rows = [dict(r) for r in records if r["authorship"] != "human_candidate"]
    rows += [dict(r) for r in spans if len(r["text"].split()) >= minimum_words]
    for row in rows:
        row["document_group"] = row["split_group"]
        row["example_granularity"] = "contiguous_span" if row["id"].startswith("span/") else "segment"
    d.assign_splits(rows)
    d.verify_records(rows)
    return rows, d.binary_examples(rows)


def choose_threshold(labels, scores, maximum_fpr=0.01):
    """Lowest validation threshold satisfying observed human FPR; >= comparison."""
    if not 0 <= maximum_fpr < 1:
        raise ValueError("Maximum false-positive rate must be in [0, 1)")
    negatives = sorted((float(p) for y, p in zip(labels, scores) if y == 0), reverse=True)
    if not negatives:
        raise ValueError("Validation requires labelled human translations")
    allowed = math.floor(len(negatives) * maximum_fpr)
    threshold = math.nextafter(negatives[allowed], math.inf)
    if threshold > 1:
        raise ValueError("No threshold in [0, 1] satisfies the validation false-positive target")
    return threshold


def train(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    inputs = [ROOT / "data/pilot/records.jsonl", ROOT / "data/pilot/translation_spans.jsonl"]
    expected = json.loads((ROOT / "data/pilot/manifest.json").read_text())
    for path in inputs[:1]:
        known = next(x["sha256"] for x in expected["outputs"] if x["file"] == str(path.relative_to(ROOT)))
        if nw.digest_file(path) != known:
            raise ValueError(f"Pilot checksum mismatch: {path}")
    span_manifest = json.loads((ROOT / "data/pilot/spans.manifest.json").read_text())
    # Verify span records using their own saved release checksum.
    span_outputs = span_manifest["outputs"]
    if isinstance(span_outputs, list):
        span_sha = next(x["sha256"] for x in span_outputs if Path(x["file"]).name == inputs[1].name)
    else:
        span_sha = span_outputs[inputs[1].name]
    if nw.digest_file(inputs[1]) != span_sha:
        raise ValueError("Span checksum mismatch")
    records, binary = prepare(d.read_jsonl(inputs[0]), d.read_jsonl(inputs[1]), args.minimum_words)
    output.mkdir(parents=True, exist_ok=True)
    nw.save_json(output / "licenses.snapshot.json", json.loads((ROOT / "config/licenses.json").read_text()))
    d.write_jsonl(output / "records.jsonl", records)
    d.write_jsonl(output / "binary_examples.jsonl", binary)
    subsets = {split: [r for r in binary if r["split"] == split] for split in ("train", "validation", "test")}
    for split, rows in subsets.items():
        if {r["binary_target"] for r in rows} != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=75000,
                                 sublinear_tf=True, dtype=np.float32, strip_accents=None)
    training = subsets["train"]
    features = vectorizer.fit_transform([r["text"] for r in training])
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, [r["binary_target"] for r in training])
    print(f"Fitted text-only baseline: {len(training):,} train examples, {features.shape[1]:,} features", flush=True)
    scores = []
    for split in ("validation", "test"):
        rows = subsets[split]
        probabilities = model.predict_proba(vectorizer.transform([r["text"] for r in rows]))[:, 1]
        scores += [{"id": row["id"], "score": float(p)} for row, p in zip(rows, probabilities)]
    by_id = {r["id"]: r for r in binary}
    validation_scores = [s for s in scores if by_id[s["id"]]["split"] == "validation"]
    threshold = choose_threshold([by_id[s["id"]]["binary_target"] for s in validation_scores],
                                 [s["score"] for s in validation_scores], args.maximum_validation_fpr)
    reports = {split: d.evaluate(binary, scores, threshold, split) for split in ("validation", "test")}
    long_examples = [r for r in binary if r["example_granularity"] == "contiguous_span"]
    long_ids = {r["id"] for r in long_examples}
    long_scores = [s for s in scores if s["id"] in long_ids]
    reports["large_passages"] = {split: d.evaluate(long_examples, long_scores, threshold, split) for split in ("validation", "test")}
    # Serialized models are locally generated artifacts; do not load untrusted pickle/joblib files.
    joblib.dump({"vectorizer": vectorizer, "classifier": model, "threshold": threshold}, output / "baseline.joblib")
    d.write_jsonl(output / "scores.jsonl", scores)
    nw.save_json(output / "evaluation.json", reports)
    names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    nw.save_json(output / "feature_audit.json", {
        "most_machine_associated": [{"feature": names[i], "coefficient": float(weights[i])} for i in np.argsort(weights)[-30:][::-1]],
        "most_human_associated": [{"feature": names[i], "coefficient": float(weights[i])} for i in np.argsort(weights)[:30]],
        "interpretation": "Associations on this benchmark can reflect its style/topics and are not proof of machine generation"})
    google = [r for r in binary if r["engine"] == "Google Translate"]
    manifest = {
        "processing_complete": True, "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [{"path": str(p), "sha256": nw.digest_file(p)} for p in inputs],
        "script_sha256": nw.digest_file(Path(__file__)), "datawork_script_sha256": nw.digest_file(ROOT / "datawork.py"),
        "dependency_versions": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "joblib")},
        "binary_examples": len(binary), "split_counts": {k: len(v) for k, v in subsets.items()},
        "large_passage_minimum_words": args.minimum_words, "large_passage_examples": len(long_examples),
        "large_passage_split_labels": {split: dict(collections.Counter(r["authorship"] for r in long_examples if r["split"] == split)) for split in subsets},
        "explicit_google_translate_examples": len(google), "explicit_google_translate_large_examples": sum(r["example_granularity"] == "contiguous_span" for r in google),
        "threshold": threshold, "maximum_observed_validation_human_fpr": args.maximum_validation_fpr,
        "features": "English target text only; character 3-5 gram TF-IDF fitted on train only",
        "limitations": ["Research baseline only; not applied to candidate corpus and not calibrated on original English news",
                        "Human controls are translations, not verified original English",
                        "Existing aligned spans were formed with a demo tokenizer, then selected by word count; no new passages synthesized",
                        "Few independent large human passages; uncertainty must use document clusters",
                        "Held-out documents share generators and domains with train; no unseen-generator guarantee",
                        "Observed validation FPR is not a bound on deployment FPR; threshold frozen before test evaluation",
                        "No long explicitly named Google Translate controls in current downloads",
                        "Identical target texts with conflicting workflow labels are excluded"]}
    manifest["outputs"] = {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    nw.save_json(output / "manifest.json", manifest)
    print(json.dumps({"binary_examples": len(binary), "large_examples": len(long_examples),
                      "threshold": threshold, "test": reports["test"]["metrics"].get("all"),
                      "large_test": reports["large_passages"]["test"]["metrics"].get("all")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/detector-v1")
    parser.add_argument("--minimum-words", type=int, default=400)
    parser.add_argument("--maximum-validation-fpr", type=float, default=0.01)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
