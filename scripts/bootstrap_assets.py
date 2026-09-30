"""Copy a trusted detector, ingest provided catalogs, and stage sample photos locally."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

from prelabel.app import Settings
from prelabel.catalog import ingest_catalog
from prelabel.files import is_image, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--example', type=Path, required=True)
    parser.add_argument('--home', type=Path, default=Path('runtime'))
    args = parser.parse_args()
    root = args.home.resolve()
    models = root / 'models'
    models.mkdir(parents=True, exist_ok=True)
    checkpoint = models / 'sku110k-2.pt'
    shutil.copyfile(args.assets / 'sku110k-2.pt', checkpoint)
    settings = Settings(checkpoint=str(checkpoint))
    write_json(root / 'settings.json', settings.model_dump())
    summaries = []
    for source in sorted(args.assets.glob('*.zip')):
        cat = ingest_catalog(source, root / 'catalogs')
        summaries.append({'source': source.name, 'catalog_id': cat['catalog_id'], 'records': len(cat['records']),
                          'references': sum(len(r['references']) for r in cat['records']), 'warnings': cat['warnings']})
    samples = root / 'sample_images'
    samples.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.example) as z:
        photos = [n for n in z.namelist() if is_image(n)]
        for number in (0, len(photos)//2, len(photos)-1):
            source = photos[number]
            (samples / Path(source).name).write_bytes(z.read(source))
    write_json(root / 'asset_inventory.json', {'catalogs': summaries, 'example_image_count': len(photos),
                                            'samples': str(samples), 'checkpoint': str(checkpoint)})
    print(json.dumps({'catalogs': summaries, 'example_image_count': len(photos), 'samples': str(samples)}, indent=2))


if __name__ == '__main__':
    main()
