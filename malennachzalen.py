import argparse, os, math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure, morphology

# ------------------ Amsterdam-Palette (Name, Code, approx. RGB) ------------------
AMSTERDAM_COLORS: List[Tuple[str,int,Tuple[int,int,int]]] = [
    ("Titanium White", 105, (245,245,245)),
    ("Titan Buff", 290, (233,226,209)),
    ("Naples Yellow Red Light", 224, (234,183,108)),
    ("Yellow Ochre", 227, (204,159,74)),
    ("Raw Sienna", 234, (186,141,77)),
    ("Burnt Sienna", 411, (138,77,53)),
    ("Raw Umber", 408, (96,80,62)),
    ("Burnt Umber", 409, (84,60,45)),
    ("Vandyke Brown", 403, (63,50,40)),
    ("Light Oxide Red", 339, (173,84,67)),
    ("Venetian Rose", 318, (204,140,132)),
    ("Quinacridone Rose", 366, (184,56,84)),
    ("Naphthol Red Medium", 315, (190,38,43)),
    ("Azo Orange", 276, (233,96,34)),
    ("Burnt Ochre", 663, (153,85,54)),
    ("Greenish Yellow", 621, (195,205,70)),
    ("Permanent Green Deep", 619, (9,86,44)),
    ("Sap Green", 623, (62,106,60)),
    ("Phthalo Green", 675, (17,84,62)),
    ("Cobalt Blue (Hue)", 512, (55,105,170)),
    ("Ultramarine", 504, (30,60,130)),
    ("Prussian Blue (Phthalo)", 566, (10,38,76)),
    ("Phthalo Blue", 570, (18,57,120)),
    ("Cerulean Blue (Hue)", 534, (90,150,190)),
    ("Sky Blue (Light)", 551, (162,200,230)),
    ("Payne’s Grey", 708, (82,95,107)),
    ("Neutral Grey", 710, (140,140,140)),
    ("Davy’s Grey", 707, (115,120,110)),
    ("Ivory Black", 701, (23,23,23)),
    ("Lamp Black", 702, (8,8,8)),
    ("Vanadium Yellow (Hue)", 269, (240,205,40)),
    ("Pale Rose", 361, (235,180,180)),
    ("Yellowish Green", 617, (145,172,60)),
]

# ------------------ Helpers ------------------

def load_font(size=3):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
        "/Library/Fonts/Arial.ttf",                          # macOS
        "C:/Windows/Fonts/arial.ttf",                        # Windows
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def center_crop_to_ratio(img: Image.Image, target_ratio=3/2):
    w, h = img.size
    r = w / h
    if abs(r - target_ratio) < 1e-3:
        return img
    if r > target_ratio:
        nw = int(h * target_ratio)
        x0 = (w - nw) // 2
        return img.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(w / target_ratio)
        y0 = (h - nh) // 2
        return img.crop((0, y0, w, y0 + nh))

def quantize_image(img: Image.Image, k=20) -> Tuple[Image.Image, np.ndarray]:
    rgb = img.convert("RGB")
    # Pillow quantize (MedianCut); dither off für klare Flächen
    q = rgb.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[:k*3]
    palette = np.array(pal, dtype=np.uint8).reshape(-1,3)
    return q.convert("P"), palette

def compute_boundaries(label_img: Image.Image) -> np.ndarray:
    arr = np.array(label_img)
    h, w = arr.shape
    boundary = np.zeros((h,w), dtype=bool)
    for dx, dy in [(1,0),(0,1),(-1,0),(0,-1)]:
        shifted = np.roll(arr, shift=dx, axis=1) if dx else np.roll(arr, shift=dy, axis=0)
        boundary |= (arr != shifted)
    return boundary

def map_palette_to_amsterdam(palette: np.ndarray):
    named_rgbs = np.array([rgb for (_,_,rgb) in AMSTERDAM_COLORS], dtype=np.float32)
    mapping = []
    for i, rgb in enumerate(palette):
        diff = named_rgbs - rgb.astype(np.float32)
        idx = int(np.argmin(np.sqrt((diff**2).sum(axis=1))))
        name, code, nrgb = AMSTERDAM_COLORS[idx]
        mapping.append((i+1, name, code, tuple(nrgb)))
    return mapping  # [(num, name, code, rgb)]

def merge_small_regions(arr: np.ndarray, min_area: int = 80) -> np.ndarray:
    """Verschmilzt sehr kleine 'Inseln' mit Nachbarflächen: pro Farbe labeln, kleine Komponenten auffüllen."""
    out = arr.copy()
    labels = np.unique(arr)
    for lab in labels:
        m = (out == lab)
        # Kleine Komponenten finden
        lab_img = measure.label(m, connectivity=1)
        props = measure.regionprops(lab_img)
        for p in props:
            if p.area < min_area:
                # Morphologisches Closing und dann weich zu nächster Umgebung mappen
                rr, cc = zip(*p.coords)
                out[tuple(np.array((rr,cc)))] = _nearest_neighbor_label(out, p.coords, exclude=lab)
    return out

def _nearest_neighbor_label(arr: np.ndarray, coords: List[Tuple[int,int]], exclude: int):
    """Wählt die häufigste Nachbarfarbe um die Koordinaten (ohne 'exclude')."""
    h, w = arr.shape
    neigh = []
    for y, x in coords:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            yy, xx = y+dy, x+dx
            if 0 <= yy < h and 0 <= xx < w:
                if arr[yy, xx] != exclude:
                    neigh.append(arr[yy, xx])
    if not neigh:
        return exclude
    return Counter(neigh).most_common(1)[0][0]

def place_numbers_smart(arr: np.ndarray, tile_large=120, tile_small=64, thresh=0.72):
    """Große Flächen: Zahl mehrfach (grobes Raster), kleine Flächen: ein centroid."""
    positions = []

    # 1) Einmal pro zusammenhängender Region (kleine Flächen abgedeckt)
    for lab in np.unique(arr):
        labeled = measure.label(arr == lab, connectivity=1)
        props = measure.regionprops(labeled)
        for p in props:
            y, x = p.centroid
            positions.append((int(x), int(y), int(lab)))

    # 2) Zusätzlich Raster über das Bild legen; wo homogen, Zahl ergänzen (für große Flächen mehrfach)
    h, w = arr.shape
    for tile in [tile_large, tile_small]:
        for y in range(tile//2, h, tile):
            for x in range(tile//2, w, tile):
                y0, x0 = max(0, y - tile//2), max(0, x - tile//2)
                y1, x1 = min(h, y + tile//2), min(w, x + tile//2)
                block = arr[y0:y1, x0:x1]
                vals, cnts = np.unique(block, return_counts=True)
                idx = np.argmax(cnts)
                if cnts[idx] / block.size >= thresh:
                    positions.append((x, y, int(vals[idx])))

    # Entdoppeln in der Nähe (Zahlen-Clustering)
    dedup = []
    seen = set()
    for x, y, lab in positions:
        key = (lab, x//24, y//24)  # grobe Zelle
        if key not in seen:
            seen.add(key)
            dedup.append((x, y, lab))
    return dedup

def render_template(arr: np.ndarray, palette_map: List[Tuple[int,str,int,Tuple[int,int,int]]], font_size=6):
    h, w = arr.shape
    # Konturen
    boundary = compute_boundaries(Image.fromarray(arr.astype(np.uint8), mode="L"))
    canvas = Image.new("L", (w, h), 255)
    # dünne Linien zeichnen
    boundary_img = Image.fromarray(np.where(boundary, 0, 255).astype(np.uint8), "L")
    canvas = Image.composite(boundary_img, canvas, Image.fromarray(boundary.astype(np.uint8)*255))
    draw = ImageDraw.Draw(canvas)

    # Zahlen
    positions = place_numbers_smart(arr)
    font = load_font(size=font_size)
    # Label (0..k-1) → Nummer (1..k)
    index_to_num = {i: i+1 for i in range(len(palette_map))}
    for x, y, lab in positions:
        num = index_to_num.get(lab, lab+1)
        s = str(num)
        bbox = draw.textbbox((0, 0), s, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw//2, y - th//2), s, fill=0, font=font)

    return canvas.convert("RGB")

def render_legend(palette_map: List[Tuple[int,str,int,Tuple[int,int,int]]], cols=2):
    rows = len(palette_map)
    col_w = 1200
    row_h = 72
    pad = 40
    width = col_w*cols + pad*2
    height = pad*2 + row_h*((rows + cols - 1)//cols) + 60
    img = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)
    big = load_font(30)
    small = load_font(24)

    header = "Farblegende – Amsterdam Standard Series"
    h_bbox = draw.textbbox((0,0), header, font=big)
    hw, hh = h_bbox[2] - h_bbox[0], h_bbox[3] - h_bbox[1]
    draw.text(((width-hw)//2, 10), header, fill=(0,0,0), font=big)

    per_col = (rows + cols - 1)//cols
    start_y = 60
    for idx, (num, name, code, rgb) in enumerate(palette_map):
        col = idx // per_col
        row = idx % per_col
        x0 = pad + col*col_w
        y0 = start_y + row*row_h

        # Zahl
        draw.text((x0, y0), f"{num:>2}", fill=(0,0,0), font=big)
        # Swatch
        sx, sy = x0 + 70, y0 + 12
        draw.rectangle([sx, sy, sx+100, sy+45], fill=rgb, outline=(0,0,0))
        # Name + Code
        draw.text((sx+120, y0+5), f"{name}", fill=(0,0,0), font=big)
        draw.text((sx+120, y0+38), f"({code})", fill=(90,90,90), font=small)

    return img

# ------------------ Pipeline ------------------

def build_pdf(
    input_path: str,
    out_pdf: str,
    colors: int = 20,
    work_w: int = 1800,
    work_h: int = 1200,
    min_region: int = 80,
    font_size: int = 6,
):
    if not os.path.exists(input_path):
        print(f"❌ Fehler: Die Eingabedatei '{input_path}' wurde nicht gefunden.")
        return
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"❌ Fehler: Die Datei '{input_path}' konnte nicht als Bild geöffnet werden: {e}")
        return
    img = center_crop_to_ratio(img, 3/2).resize((work_w, work_h), Image.Resampling.LANCZOS)

    # Quantisieren
    label_img, pal = quantize_image(img, k=colors)
    arr = np.array(label_img, dtype=np.uint8)

    # Kleine Regionen verschmelzen
    arr = merge_small_regions(arr, min_area=min_region)

    # Palette → Amsterdam-Mapping
    palette_map = map_palette_to_amsterdam(pal)

    # Seite 1: Vorlage
    page1 = render_template(arr, palette_map, font_size=font_size)

    # Seite 2: Legende
    page2 = render_legend(palette_map, cols=2)

    # PDF speichern in 60×40 cm @300dpi
    dpi = 300
    target_px = (int(60/2.54*dpi), int(40/2.54*dpi))  # ~7087×4724
    page1_big = page1.resize(target_px, Image.Resampling.NEAREST)  # NEAREST = klare Linien/Zahlen
    page2_w = page2.width
    scale = target_px[0] / page2_w
    page2_big = page2.resize((target_px[0], int(page2.height*scale)), Image.Resampling.NEAREST)

    page1_big.save(out_pdf, "PDF", resolution=dpi, save_all=True, append_images=[page2_big])
    print(f"✅ PDF gespeichert: {out_pdf}")

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="Malen-nach-Zahlen PDF Generator")
    ap.add_argument("input", help="Eingabebild (JPG/PNG)")
    ap.add_argument("--out", default="malen_nach_zahlen.pdf", help="Ausgabe-PDF")
    ap.add_argument("--colors", type=int, default=30, help="Anzahl Farben (z.B. 20)")
    ap.add_argument("--width", type=int, default=1800, help="Arbeitsbreite in px")
    ap.add_argument("--height", type=int, default=1200, help="Arbeitshöhe in px")
    ap.add_argument("--min-region", type=int, default=80, help="Min. Flächengröße für Verschmelzung (Pixel)")
    ap.add_argument("--font-size", type=int, default=7, help="Zahlengröße")
    args = ap.parse_args()

    build_pdf(
        input_path=args.input,
        out_pdf=args.out,
        colors=args.colors,
        work_w=args.width,
        work_h=args.height,
        min_region=args.min_region,
        font_size=args.font_size,
    )

if __name__ == "__main__":
    main()
