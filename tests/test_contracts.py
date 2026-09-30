import io
import zipfile
import json
from pathlib import Path

import pytest
from PIL import Image

from prelabel.catalog import ingest_catalog
from prelabel.exporter import export_zip, yolo_row
from prelabel.images import prepare_image, import_images
from prelabel.policy import decide, resolve_packs


def catalog_zip(path, names=None, missing=False):
    buffer = io.BytesIO()
    Image.new('RGB', (20, 30), 'red').save(buffer, format='PNG')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('catalog.yaml', names or 'names:\n  7: "00123 - Target"\n  2: "999 - Other"\n')
        z.writestr('references/7.png', buffer.getvalue())
        if not missing:
            z.writestr('references/2.png', buffer.getvalue())


def test_catalog_uses_ids_not_archive_order(tmp_path):
    source = tmp_path / 'catalog.zip'
    catalog_zip(source)
    cat = ingest_catalog(source, tmp_path / 'out')
    assert {(r['source_id'], r['barcode']) for r in cat['records']} == {('7', '00123'), ('2', '999')}
    assert all(Path(r['references'][0]['path']).exists() for r in cat['records'])


def test_catalog_missing_reference_fails(tmp_path):
    source = tmp_path / 'catalog.zip'
    catalog_zip(source, missing=True)
    with pytest.raises(ValueError, match='reference'):
        ingest_catalog(source, tmp_path / 'out')


def test_duplicates_are_preserved_and_reported(tmp_path):
    source = tmp_path / 'catalog.zip'
    catalog_zip(source, 'names:\n  7: "00123 - Large"\n  2: "00123 - Small"\n')
    cat = ingest_catalog(source, tmp_path / 'out')
    assert len(cat['records']) == 2
    assert cat['warnings'] and cat['records'][0]['sku_id'] != cat['records'][1]['sku_id']


def test_unsafe_archive_rejected(tmp_path):
    source = tmp_path / 'evil.zip'
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('../escape.jpg', b'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        import_images(source, tmp_path / 'out')


def test_orientation_matches_export_pixels(tmp_path):
    source = tmp_path / 'rotated.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB', (40, 20)).save(source, exif=exif)
    original = source.read_bytes()
    result = prepare_image(source, tmp_path / 'out.jpg')
    assert (result['width'], result['height']) == (20, 40)
    assert result['orientation_normalized']
    assert Image.open(result['path']).size == (20, 40)
    assert source.read_bytes() == original


def test_yolo_contract_and_invalid_boxes(tmp_path):
    assert yolo_row([10, 20, 30, 60], 100, 100) == '0 0.20000000 0.40000000 0.20000000 0.40000000'
    for box in ([0, 0, 0, 1], [-1, 0, 2, 2], [0, 0, float('nan'), 1]):
        with pytest.raises(ValueError):
            yolo_row(box, 100, 100)
    image = tmp_path / 'shelf.jpg'
    Image.new('RGB', (100, 100)).save(image)
    out = export_zip([{'path': str(image), 'name': 'shelf.jpg', 'width': 100, 'height': 100, 'detections': []}], tmp_path / 'set_001.zip')
    with zipfile.ZipFile(out) as z:
        assert set(z.namelist()) == {'data.yaml', 'images/shelf.jpg', 'labels/shelf.txt'}
        assert z.read('labels/shelf.txt') == b''
        assert z.read('images/shelf.jpg') == image.read_bytes()
        assert z.read('data.yaml').decode() == 'path: .\ntrain: images\n\nnames:\n  0: product\n'


def test_uncertain_and_low_similarity_products_retained():
    for evidence in ({}, {'candidates': [{'score': -0.8}]}, {'ocr_evidence': []}):
        assert decide(evidence)['decision'] == 'retain_uncertain'
    assert decide({'verified_exclusion': {'validated': False, 'kind': 'excluded_product'}})['decision'] != 'remove'
    assert decide({'verified_exclusion': {'validated': True, 'kind': 'excluded_product', 'validation_id': 'reviewed-v1'}})['decision'] == 'remove'


def test_adjacent_boxes_never_merge_and_pack_requires_evidence():
    boxes = [{'detection_id': 'a', 'bbox_xyxy_pixels': [0, 0, 10, 20]}, {'detection_id': 'b', 'bbox_xyxy_pixels': [10, 0, 20, 20]}]
    assert resolve_packs(boxes) == boxes
    outer = {'detection_id': 'p', 'bbox_xyxy_pixels': [0, 0, 20, 20], 'pack_evidence': {'validated': True, 'validation_id': 'pack-v1', 'outer_pack': True, 'constituent_ids': ['a', 'b']}}
    resolved = resolve_packs([*boxes, outer])
    assert [b['detection_id'] for b in resolved if b.get('decision') != 'remove'] == ['p']
    assert len(resolved) == 3


def test_duplicate_image_stems_are_disambiguated(tmp_path):
    for folder in ['one', 'two']:
        (tmp_path / folder).mkdir()
        Image.new('RGB', (10, 20)).save(tmp_path / folder / 'a.jpg')
    result = import_images(tmp_path, tmp_path / 'staged')
    assert len(result) == 2
    assert len({r['name'] for r in result}) == 2


def test_zip_raw_prefix_cannot_overwrite_another_photo(tmp_path):
    red, blue = io.BytesIO(), io.BytesIO()
    Image.new('RGB', (10, 20), 'red').save(red, format='PNG')
    Image.new('RGB', (10, 20), 'blue').save(blue, format='PNG')
    source = tmp_path / 'photos.zip'
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('raw_a.png', red.getvalue())
        z.writestr('a.png', blue.getvalue())
    images = import_images(source, tmp_path / 'staged')
    assert len(images) == 2
    assert Image.open(images[0]['path']).getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(images[1]['path']).getpixel((0, 0)) == (0, 0, 255)


def test_mixed_yaml_id_types_rejected(tmp_path):
    source = tmp_path / 'catalog.zip'
    catalog_zip(source, 'names:\n  7: "00123 - Target"\n  "7": "00456 - Other"\n')
    with pytest.raises(ValueError, match='Duplicate.*ID'):
        ingest_catalog(source, tmp_path / 'out')


def test_catalog_role_is_part_of_identity(tmp_path):
    source = tmp_path / 'catalog.zip'
    catalog_zip(source)
    target = ingest_catalog(source, tmp_path / 'out', target=True)
    excluded = ingest_catalog(source, tmp_path / 'out', target=False)
    assert target['catalog_id'] != excluded['catalog_id']
    assert len(list((tmp_path / 'out').glob('*/catalog.json'))) == 2


def test_generated_image_name_cannot_collide_with_existing_filename(tmp_path):
    import hashlib
    generated = 'a_' + hashlib.sha256(b'two/a.png').hexdigest()[:10] + '.png'
    source = tmp_path / 'photos.zip'
    with zipfile.ZipFile(source, 'w') as z:
        for name, color in [('a.png', 'red'), (generated, 'blue'), ('two/a.png', 'green')]:
            buffer = io.BytesIO()
            Image.new('RGB', (10, 10), color).save(buffer, format='PNG')
            z.writestr(name, buffer.getvalue())
    images = import_images(source, tmp_path / 'staged')
    assert len({i['name'] for i in images}) == 3
    assert Image.open(images[1]['path']).getpixel((0, 0)) == (0, 0, 255)
