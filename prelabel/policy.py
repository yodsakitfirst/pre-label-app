from copy import deepcopy


def decide(evidence: dict, baseline=False, config=None, image_size=None):
    config = config or {}
    result = {'decision': 'retain_uncertain', 'decision_reason': 'unvalidated_target_membership',
              'export_class_id': 0, 'assigned_sku': None}
    verified = evidence.get('verified_exclusion', {})
    if not baseline and verified.get('validated') is True and verified.get('validation_id') and verified.get('kind') in ('excluded_product', 'non_product'):
        result.update(decision='remove', decision_reason='validated_' + verified['kind'])
    elif baseline:
        result['decision_reason'] = 'baseline_detector_box'
    if result['decision'] == 'remove':
        return result
    if config.get('full_product_only') and image_size and evidence.get('bbox_xyxy_pixels'):
        width, height = image_size
        x1, y1, x2, y2 = evidence['bbox_xyxy_pixels']
        margin = config.get('edge_margin_fraction', 0.01)
        sides = [side for side, distance in [('left', x1 / width), ('top', y1 / height),
                 ('right', (width - x2) / width), ('bottom', (height - y2) / height)] if distance <= margin]
        result['completeness_evidence'] = {'status': 'possible_edge_truncation' if sides else 'no_image_edge_risk',
                                            'image_edges': sides, 'margin_fraction': margin, 'occlusion_checked': False}
        if sides:
            result.update(decision='defer_review', decision_reason='possible_image_edge_truncation')
            return result
    if config.get('mode') == 'reference_filter':
        candidates = evidence.get('candidates', [])
        targets = [c for c in candidates if c.get('target') is True]
        excluded = [c for c in candidates if c.get('target') is False]
        target = max(targets, key=lambda c: c['score'], default=None)
        exclusion = max(excluded, key=lambda c: c['score'], default=None)
        rival = max((c for c in targets if target and c.get('barcode', c.get('sku_id')) != target.get('barcode', target.get('sku_id'))),
                    key=lambda c: c['score'], default=None)
        threshold = config.get('reference_min_similarity', 0.75)
        result['reference_gate'] = {'minimum_similarity': threshold, 'calibrated': False,
                                    'best_target': target, 'best_excluded': exclusion,
                                    'excluded_margin': config.get('excluded_margin', 0.03),
                                    'best_competing_sku': rival, 'sku_margin': config.get('reference_sku_margin', 0.03),
                                    'sku_margin_filters_export': False,
                                    'sku_identity_ambiguous': bool(target and rival and target['score'] - rival['score'] < config.get('reference_sku_margin', 0.03))}
        result.update(decision='defer_review', decision_reason='no_target_reference_candidate')
        if target and target['score'] < threshold:
            result['decision_reason'] = 'below_reference_export_threshold'
        elif target and exclusion and target['score'] - exclusion['score'] < config.get('excluded_margin', 0.03):
            result['decision_reason'] = 'target_excluded_reference_ambiguity'
        elif target:
            result.update(decision='retain', decision_reason='meets_reference_export_threshold')
    return result


def resolve_packs(detections: list[dict]) -> list[dict]:
    result = deepcopy(detections)
    by_id = {d['detection_id']: d for d in result}
    for outer in result:
        evidence = outer.get('pack_evidence', {})
        if not (evidence.get('validated') is True and evidence.get('validation_id') and evidence.get('outer_pack') is True):
            continue
        x1, y1, x2, y2 = outer['bbox_xyxy_pixels']
        for constituent in evidence.get('constituent_ids', []):
            child = by_id.get(constituent)
            if child is None or child is outer:
                continue
            a, b, c, d = child['bbox_xyxy_pixels']
            if x1 <= a < c <= x2 and y1 <= b < d <= y2:
                child.update(decision='remove', decision_reason='validated_pack_constituent', outer_pack_id=outer['detection_id'])
    return result
