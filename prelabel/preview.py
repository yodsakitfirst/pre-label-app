"""Optional drawn previews are separate artifacts, never editable export photos."""
from PIL import Image, ImageDraw


def save_preview(result, path):
    with Image.open(result['path']) as original:
        image = original.convert('RGB')
    draw = ImageDraw.Draw(image)
    for d in result['detections']:
        if d['decision'] not in ('retain', 'retain_uncertain'):
            continue
        box = d['bbox_xyxy_pixels']
        draw.rectangle(box, outline='#00bb70', width=max(2, image.width // 400))
        match = d.get('reference_gate', {}).get('best_target')
        if match:
            text = f"candidate {match['barcode']} | {match['score']:.2f}"
            x, y = box[0], max(0, box[1] - 14)
            area = draw.textbbox((x, y), text)
            draw.rectangle(area, fill='#173b34')
            draw.text((x, y), text, fill='white')
    image.thumbnail((1600, 1600))
    image.save(path, quality=92)
    return str(path)
