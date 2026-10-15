"""Recall-first, class-wise one-to-one physical-coordinate candidate evaluation.

Consumes exported candidates; never modifies semantic predictions or RELION.
Matching maximizes TP count first, then minimizes total localization distance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree


def match_particles(gt, predicted, tolerance):
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 3)
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(gt).all() or not np.isfinite(predicted).all():
        raise ValueError('Finite coordinates and positive physical distance tolerance required')
    matched_gt, matched_pred = np.empty(0, int), np.empty(0, int)
    if len(gt) and len(predicted):
        # Sparse radius edges avoid a full particle-count-squared distance matrix.
        neighbors = cKDTree(gt).query_ball_tree(cKDTree(predicted), tolerance)
        rows = np.repeat(np.arange(len(gt)), [len(n) for n in neighbors])
        columns = np.asarray([j for n in neighbors for j in n], dtype=int)
        distances = np.linalg.norm(gt[rows] - predicted[columns], axis=1)
        # Subtracting the common cost 1: dummy costs 1, all localization costs
        # together are <1. Positive costs also preserve sparse zero-distance edges.
        costs = 1 + distances / tolerance / (min(len(gt), len(predicted)) + 1)
        graph = coo_matrix((np.concatenate((costs, np.full(len(gt), 2.))),
            (np.concatenate((rows, np.arange(len(gt)))),
             np.concatenate((columns, len(predicted)+np.arange(len(gt)))))),
            shape=(len(gt), len(predicted)+len(gt))).tocsr()
        rows, columns = min_weight_full_bipartite_matching(graph)
        keep = columns < len(predicted)
        matched_gt, matched_pred = rows[keep], columns[keep]
    unmatched = np.ones(len(predicted), dtype=bool)
    unmatched[matched_pred] = False
    duplicates = int((cKDTree(gt[matched_gt]).query(predicted[unmatched])[0] <= tolerance).sum()) if len(matched_gt) else 0
    errors = np.linalg.norm(gt[matched_gt]-predicted[matched_pred], axis=1).tolist()
    tp = len(errors)
    return dict(gt_count=len(gt), candidates=len(predicted), tp=tp, fn=len(gt)-tp, fp=len(predicted)-tp,
                duplicate_candidates=duplicates, localization_errors=errors)


def summarize(rows):
    result = {key: sum(row[key] for row in rows) for key in
              ('gt_count', 'candidates', 'tp', 'fn', 'fp', 'duplicate_candidates')}
    errors = [e for row in rows for e in row['localization_errors']]
    result.update(recall=result['tp']/result['gt_count'] if result['gt_count'] else None,
                  precision=result['tp']/result['candidates'] if result['candidates'] else None,
                  miss_rate=result['fn']/result['gt_count'] if result['gt_count'] else None,
                  duplicate_rate=result['duplicate_candidates']/result['candidates'] if result['candidates'] else 0.,
                  candidates_per_gt=result['candidates']/result['gt_count'] if result['gt_count'] else None,
                  localization_mean=float(np.mean(errors)) if errors else None,
                  localization_median=float(np.median(errors)) if errors else None,
                  localization_p95=float(np.percentile(errors, 95)) if errors else None)
    return result


def evaluate_cases(cases, tolerances, threshold=None, budget=None, prediction_key='predictions'):
    if not tolerances:
        raise ValueError('Provide preselected per-class physical matching tolerances')
    rows = []
    for case in cases:
        if set(case['classes']) - set(tolerances):
            raise ValueError('Every class needs an explicit tolerance')
        if not case.get('valid_region_confirmed', False):
            raise ValueError('Candidates and GT must cover the same complete valid tomogram region')
        for name, tolerance in tolerances.items():
            entry = case['classes'].get(name, {})
            predictions = entry.get(prediction_key, [])
            if any(not np.isfinite(p['score']) for p in predictions):
                raise ValueError('Candidate confidence scores must be finite')
            selected = [p for p in predictions if threshold is None or p['score'] >= threshold]
            selected = sorted(selected, key=lambda p: p['score'], reverse=True)
            if budget is not None:
                selected = selected[:budget]
            stats = match_particles(entry.get('gt', []), [p['center'] for p in selected], tolerance)
            rows.append(dict(case_id=case['case_id'], source=case.get('source', case['case_id']),
                category=name, valid_volume=case.get('valid_volume'), input_candidates=len(predictions), **stats))
    by_class = {}
    for name in tolerances:
        subset = [row for row in rows if row['category'] == name]
        summary = summarize(subset)
        summary['fp_per_tomo'] = summary['fp'] / max(len(subset), 1)
        volumes = [row['valid_volume'] for row in subset]
        volume = sum(volumes) if volumes and all(v is not None and v > 0 for v in volumes) else None
        summary['fp_per_valid_volume'] = summary['fp'] / volume if volume else None
        summary['candidates_per_valid_volume'] = summary['candidates'] / volume if volume else None
        by_class[name] = summary
    recalls = [v['recall'] for v in by_class.values() if v['recall'] is not None]
    return dict(threshold=threshold, per_tomo_per_class_candidate_budget=budget, classes=by_class,
                macro_recall=float(np.mean(recalls)) if recalls else None, cases=rows)


def bootstrap_macro_recall(cases, tolerances, repetitions, seed=0):
    groups = sorted({case.get('source', case['case_id']) for case in cases})
    if len(groups) < 2:
        return {'status': 'insufficient_independent_sources'}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        selected = [case for group in rng.choice(groups, len(groups)) for case in cases
                    if case.get('source', case['case_id']) == group]
        value = evaluate_cases(selected, tolerances)['macro_recall']
        if value is not None:
            values.append(value)
    return dict(unit='source_or_tomogram', repetitions=repetitions, seed=seed,
                ci95=np.percentile(values, [2.5, 97.5]).tolist() if values else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSON with coordinate_units, tolerances, cases')
    parser.add_argument('--output', required=True)
    parser.add_argument('--thresholds', nargs='*', type=float, default=[])
    parser.add_argument('--budgets', nargs='*', type=int, default=[])
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    if not payload.get('coordinate_units') or payload['coordinate_units'] == 'voxel':
        raise ValueError('Convert coordinates to explicit physical units before evaluation')
    if any(b < 0 for b in args.budgets):
        raise ValueError('Candidate budgets must be nonnegative')
    cases, tolerances = payload['cases'], payload['tolerances']
    report = dict(coordinate_units=payload['coordinate_units'], tolerances=tolerances,
        baseline=evaluate_cases(cases, tolerances),
        threshold_curve=[evaluate_cases(cases, tolerances, threshold=t) for t in args.thresholds],
        candidate_budget_curve=[evaluate_cases(cases, tolerances, budget=b) for b in args.budgets])
    if any('predictions_before' in entry for case in cases for entry in case['classes'].values()):
        if not all('predictions_before' in case['classes'].get(name, {}) for case in cases for name in tolerances):
            raise ValueError('Before/after comparison requires predictions_before for every case/class')
        report['before_processing'] = evaluate_cases(cases, tolerances, prediction_key='predictions_before')
    if args.bootstrap:
        report['recall_bootstrap'] = bootstrap_macro_recall(cases, tolerances, args.bootstrap)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
