"""Auditable OCR/visual membership fusion; raw similarities are not probabilities."""
import re

from .ocr import normalize_text

VERSION = 'ocr_visual_membership_v1'
BRANDS = {
    'sunsilk': ('sunsilk', 'ซันซิล', 'ซันซิลค์'),
    'clear': ('clear', 'เคลียร์'), 'dove': ('dove', 'โดฟ'),
    'tresemme': ('tresemme', 'tresemme', 'เทรซาเม่', 'เทรเซมเม่'),
    'pantene': ('pantene', 'แพนทีน'), 'head shoulders': ('head shoulders', 'head and shoulders', 'เฮดแอนด์โชว์เดอร์'),
    'loreal': ('loreal', 'l oreal', 'ลอรีอัล'), 'elseve': ('elseve', 'เอลแซฟ'),
    'tsubaki': ('tsubaki', 'ซึบากิ'), 'kerasys': ('kerasys', 'เคราซิส'),
    'herbal essences': ('herbal essences', 'เฮอร์บัล'),
    'rejoice': ('rejoice', 'รีจอยส์'), 'lux': ('lux', 'ลักส์'),
    'vaseline': ('vaseline', 'วาสลีน'), 'ponds': ('ponds', 'พอนด์ส'),
}
STOP = {'shampoo', 'conditioner', 'product', 'bottle', 'pack', 'new', 'the', 'and', 'for', 'ml',
        'hair', 'care', 'สูตร', 'แชมพู', 'ครีมนวด', 'ครีมนวดผม', 'ผม', 'ใหม่', 'ขวด', 'มล'}


def brands(text):
    text = normalize_text(text)
    return {brand for brand, aliases in BRANDS.items()
            if any(alias in text if re.search('[ก-๙]', alias)
                   else re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text)
                   for alias in aliases)}


def prepare(evidence, config):
    lines = [{'text': str(r.get('text', '')), 'score': float(r['recognition_score'])}
             for r in evidence.get('ocr_evidence', [])
             if r.get('recognition_score', 0) >= config.get('ocr_min_score', .7)]
    text = normalize_text(' '.join(r['text'] for r in lines))
    observed = brands(text)
    numbers = set(re.findall(r'\d{8,}', text))
    candidates = evidence.get('candidates', [])
    targets = [c for c in candidates if c.get('target') is True]
    # Use all selected target records, not just the short visual list, for brand rejection.
    records = [r for cat in config.get('catalogs', []) for r in cat.get('records', [])
               if r.get('target', cat.get('target', True)) is True]
    target_names = records or targets
    target_brands = set().union(*(brands(c.get('name', '')) for c in target_names))
    complete_brand_coverage = bool(target_names) and all(brands(c.get('name', '')) for c in target_names)
    barcode_hits = [c for c in candidates if str(c.get('barcode', '')) in numbers]
    positive = any(c.get('target') is True for c in barcode_hits)
    negative = bool(barcode_hits) and all(c.get('target') is False for c in barcode_hits)
    conflict = bool(complete_brand_coverage and observed and observed.isdisjoint(target_brands))
    tokens = {t for t in text.split() if len(t) >= 3 and not t.isdecimal()} - STOP
    brand_words = {word for brand in observed for alias in BRANDS[brand] for word in alias.split()}
    supported = []
    for c in targets:
        name = normalize_text(c.get('name', ''))
        # Brand alone cannot rescue a low-scoring variant or size.
        extra = {t for t in tokens.intersection(name.split()) - brand_words if not brands(t)}
        if observed.intersection(brands(name)) and len(extra) >= 2:
            supported.append(c)
    useful = bool(observed or barcode_hits or supported)
    rule = 'include' if positive or supported else 'exclude' if negative or conflict else 'uncertain'
    # English aliases supplement Thai names for the published English Laya checkpoint.
    ordered = sorted(candidates, key=lambda c: (str(c.get('barcode', '')) in numbers, c['score']), reverse=True)
    shortlist = ordered[:5]
    for group in (targets, [c for c in candidates if c.get('target') is False]):
        if group and not any(c.get('target') == group[0].get('target') for c in shortlist):
            shortlist[-1:] = [max(group, key=lambda c: c['score'])]
    state = {'ocr': lines, 'observed_brands': sorted(observed), 'all_target_brands': sorted(target_brands),
             'visual_cutoff': config.get('reference_min_similarity', .75),
             'target_brand_coverage_complete': complete_brand_coverage,
             'candidates': [{'name': c.get('name', ''), 'brand': sorted(brands(c.get('name', ''))),
                             'barcode': c.get('barcode'), 'target': c.get('target'),
                             'similarity': round(float(c['score']), 4)} for c in shortlist],
             'text_supports_target': positive or bool(supported), 'text_conflicts_with_targets': negative or conflict,
             'scores_are_uncalibrated': True}
    support = [c for c in targets if c in barcode_hits or c in supported]
    return state, useful, rule, support


def fuse(evidence, visual, config, engine=None):
    backend = config.get('fusion_backend', 'visual')
    if backend == 'visual' or visual.get('decision_reason') == 'possible_image_edge_truncation' or visual.get('decision_reason', '').startswith('validated_'):
        return visual
    state, useful, rule, support = prepare(evidence, config)
    audit = {'version': VERSION, 'backend': backend, 'visual_decision': visual['decision'],
             'simple_rule': rule, 'status': 'no_useful_text', 'applied': False, 'state': state,
             'probabilities_calibrated_for_catalogs': False}
    result = {**visual, 'fusion_evidence': audit}
    if not useful:
        return result
    choice, probability = rule, 1.0
    if backend == 'laya':
        if engine is None:
            audit['status'] = 'unavailable'
            return result
        try:
            output = engine.predict(state)
            audit.update(status='completed', laya=output)
            choice = output.get('choice')
            probability = output.get('probabilities', {}).get(choice, 0)
            if choice not in ('include', 'exclude', 'uncertain') or not isinstance(probability, (float, int)) or not 0 <= probability <= 1:
                raise ValueError('Invalid Laya membership answer')
        except Exception as exc:
            audit.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            return result
    else:
        audit['status'] = 'completed'
    # The model cannot invent evidence for exclusion or rescue. Disagreement falls back.
    if probability < config.get('fusion_min_probability', .9) or choice != rule:
        return result
    if choice == 'exclude':
        result.update(decision='defer_review', decision_reason='ocr_catalog_conflict')
    elif choice == 'include' and support:
        floor = config.get('reference_min_similarity', .75) - .05
        if max(c['score'] for c in support) >= floor:
            result.update(decision='retain', decision_reason='ocr_visual_catalog_support')
    audit['applied'] = result['decision'] != visual['decision']
    return result
