from copy import deepcopy


def decide(evidence: dict, baseline=False):
    result = {'decision': 'retain_uncertain', 'decision_reason': 'unvalidated_target_membership',
              'export_class_id': 0, 'assigned_sku': None}
    verified = evidence.get('verified_exclusion', {})
    if not baseline and verified.get('validated') is True and verified.get('validation_id') and verified.get('kind') in ('excluded_product', 'non_product'):
        result.update(decision='remove', decision_reason='validated_' + verified['kind'])
    elif baseline:
        result['decision_reason'] = 'baseline_detector_box'
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
