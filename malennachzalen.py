import argparse, os, sys, json, csv, yaml
from collections import Counter
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from skimage import measure

# Optional SciPy
try:
    from scipy.ndimage import maximum_filter
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    linear_sum_assignment = None
    maximum_filter = None

# Optional Bilateral-Filter
try:
    from skimage.restoration import denoise_bilateral
    HAVE_BILATERAL = True
except Exception:
    HAVE_BILATERAL = False

# ==========================================================
# Fortschritt
# ==========================================================
def _bar(pct, width=32):
    pct = max(0, min(100, float(pct)))
    filled = int(width * pct / 100.0)
    rem = max(0, width - filled)
    head = ">" if rem > 0 else ""
    return "[" + "="*filled + head + "."*(max(0, rem-1)) + "]"

def progress(pct, msg):
    pct = max(0, min(100, int(round(pct))))
    print(f"{_bar(pct)} {pct:>3}%  {msg}")
    sys.stdout.flush()

# ==========================================================
# Font / Utils
# ==========================================================
def load_font(size=8):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/Library/Fonts/Arial.ttf",                         # macOS
        "C:/Windows/Fonts/arial.ttf",                       # Windows
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def hex_to_rgb(hexstr: str) -> Tuple[int,int,int]:
    s = hexstr.strip().lstrip("#")
    if len(s) == 3: s = "".join([c*2 for c in s])
    if len(s) != 6: raise ValueError(f"Invalid HEX: {hexstr}")
    return tuple(int(s[i:i+2], 16) for i in (0,2,4))

# ==========================================================
# Palette laden
# ==========================================================
def load_palette_csv(path: str) -> List[Tuple[str,int,Tuple[int,int,int]]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("Name") or ""
            code = int(row.get("code") or row.get("Code"))
            if "hex" in row and row["hex"]:
                rgb = hex_to_rgb(row["hex"])
            else:
                r = int(row.get("r") or row.get("R"))
                g = int(row.get("g") or row.get("G"))
                b = int(row.get("b") or row.get("B"))
                rgb = (r,g,b)
            rows.append((name, code, rgb))
    if not rows:
        raise ValueError("Palette file is empty")
    return rows

# ==========================================================
# Presets laden
# ==========================================================
def load_presets(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================================
# Bildvorbereitung / Quantisierung
# ==========================================================
def center_crop_to_ratio(img: Image.Image, target_ratio=3/2):
    w, h = img.size; r = w / h
    if abs(r - target_ratio) < 1e-3: return img
    if r > target_ratio:
        nw = int(h * target_ratio); x0 = (w - nw) // 2
        return img.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(w / target_ratio); y0 = (h - nh) // 2
        return img.crop((0, y0, w, y0 + nh))

def preprocess_image(img: Image.Image,
                     do_autocontrast=True, gamma=0.9,
                     color_boost=1.05, bilateral=False):
    out = img
    if do_autocontrast:
        out = ImageOps.autocontrast(out, cutoff=1)
    if abs(gamma - 1.0) > 1e-3:
        lut = [int(((i/255.0)**gamma)*255 + 0.5) for i in range(256)]
        out = out.point(lut*3)
    if abs(color_boost - 1.0) > 1e-3:
        out = ImageEnhance.Color(out).enhance(color_boost)
    if bilateral and HAVE_BILATERAL:
        arr = np.asarray(out, dtype=np.float32)/255.0
        arr = denoise_bilateral(arr, sigma_color=0.06, sigma_spatial=3, channel_axis=2)
        out = Image.fromarray((arr*255).clip(0,255).astype(np.uint8), 'RGB')
    return out

def quantize_image(img: Image.Image, k=20):
    rgb = img.convert("RGB")
    q = rgb.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[:k*3]
    palette = np.array(pal, dtype=np.uint8).reshape(-1,3)
    return q.convert("P"), palette

# ==========================================================
# Konturen
# ==========================================================
def compute_boundaries_no_wrap(label_img: Image.Image) -> np.ndarray:
    arr = np.array(label_img); h, w = arr.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, 1:] |= (arr[:, 1:] != arr[:, :-1]); boundary[:, :-1] |= (arr[:, :-1] != arr[:, 1:])
    boundary[1:, :] |= (arr[1:, :] != arr[:-1, :]); boundary[:-1, :] |= (arr[:-1, :] != arr[1:, :])
    return boundary

# ==========================================================
# Farbzuordnung
# ==========================================================
def map_palette_to_amsterdam(palette: np.ndarray,
                             amsterdam: List[Tuple[str,int,Tuple[int,int,int]]]):
    named_rgbs = np.array([rgb for (_,_,rgb) in amsterdam], dtype=np.float32)
    mapping = []
    for i, rgb in enumerate(palette):
        diff = named_rgbs - rgb.astype(np.float32)
        idx = int(np.argmin(np.sqrt((diff**2).sum(axis=1))))
        name, code, nrgb = amsterdam[idx]
        mapping.append((i+1, name, code, tuple(nrgb)))
    return mapping

def assign_unique_amsterdam_weighted(palette: np.ndarray,
                                     label_img_arr: np.ndarray,
                                     amsterdam: List[Tuple[str,int,Tuple[int,int,int]]]):
    src = palette.astype(np.float32)
    labels, counts = np.unique(label_img_arr, return_counts=True)
    k = src.shape[0]
    weights = np.ones(k, dtype=np.float32); weights[labels] = counts.astype(np.float32)
    weights /= weights.max() + 1e-6

    am_names = [n for (n,_,_) in amsterdam]
    am_codes = [c for (_,c,_) in amsterdam]
    am_rgbs  = np.array([rgb for (*_, rgb) in amsterdam], dtype=np.float32)

    D = np.sqrt(((src[:,None,:] - am_rgbs[None,:,:])**2).sum(axis=2))
    row_factors = 0.5 + 1.5*weights.reshape(-1,1)
    cost = D * row_factors

    np.random.seed(0)
    if HAVE_SCIPY and linear_sum_assignment is not None and cost.shape[1] >= cost.shape[0]:
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = []
        for i, j in zip(row_ind, col_ind):
            name, code = am_names[j], am_codes[j]
            rgb = tuple(map(int, am_rgbs[j]))
            mapping.append((i+1, name, code, rgb))
        mapping.sort(key=lambda t: t[0])
        return mapping

    used = set(); mapping = []
    for i in range(src.shape[0]):
        order = np.argsort(cost[i])
        j = next((x for x in order if x not in used), order[0])
        used.add(j)
        name, code = am_names[j], am_codes[j]
        rgb = tuple(map(int, am_rgbs[j]))
        mapping.append((i+1, name, code, rgb))
    return mapping

# ==========================================================
# Flächen-Bereinigung & Nummern
# ==========================================================
def _nearest_neighbor_label(arr: np.ndarray, coords, exclude: int):
    h, w = arr.shape; neigh = []
    for y, x in coords:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            yy, xx = y+dy, x+dx
            if 0 <= yy < h and 0 <= xx < w and arr[yy, xx] != exclude:
                neigh.append(arr[yy, xx])
    return (Counter(neigh).most_common(1)[0][0]) if neigh else exclude

def merge_small_regions(arr: np.ndarray, min_area: int = 120) -> np.ndarray:
    out = arr.copy()
    for lab in np.unique(arr):
        m = (out == lab); labeled = measure.label(m, connectivity=1)
        for p in measure.regionprops(labeled):
            if p.area < min_area:
                rr, cc = p.coords[:,0], p.coords[:,1]
                out[rr, cc] = _nearest_neighbor_label(out, p.coords, exclude=lab)
    return out

def place_numbers_smart(arr: np.ndarray, tile_large=100, tile_small=56, thresh=0.72):
    positions = []
    for lab in np.unique(arr):
        labeled = measure.label(arr == lab, connectivity=1)
        for p in measure.regionprops(labeled):
            y, x = p.centroid; positions.append((int(x), int(y), int(lab)))
    h, w = arr.shape
    for tile in [tile_large, tile_small]:
        for y in range(tile//2, h, tile):
            for x in range(tile//2, w, tile):
                y0, x0 = max(0, y - tile//2), max(0, x - tile//2)
                y1, x1 = min(h, y + tile//2), min(w, x + tile//2)
                block = arr[y0:y1, x0:x1]
                vals, cnts = np.unique(block, return_counts=True)
                idx = int(np.argmax(cnts))
                if cnts[idx] / block.size >= thresh:
                    positions.append((x, y, int(vals[idx])))
    dedup, seen = [], set()
    for x, y, lab in positions:
        key = (lab, x//24, y//24)
        if key not in seen:
            seen.add(key); dedup.append((x, y, lab))
    return dedup

# ==========================================================
# Rendering
# ==========================================================
def render_template(arr, palette_map, font_size=8, thicken=False):
    boundary = compute_boundaries_no_wrap(Image.fromarray(arr.astype(np.uint8)))
    if thicken and HAVE_SCIPY and maximum_filter is not None:
        boundary = maximum_filter(boundary.astype(np.uint8), size=2) > 0
    canvas = Image.new("L", arr.shape[::-1], 255)
    boundary_img = Image.fromarray(np.where(boundary, 0, 255).astype(np.uint8))
    canvas = Image.composite(boundary_img, canvas, Image.fromarray((boundary>0).astype(np.uint8)*255))
    draw = ImageDraw.Draw(canvas)
    positions = place_numbers_smart(arr)
    font = load_font(size=font_size)
    index_to_num = {i: i+1 for i in range(len(palette_map))}
    for x, y, lab in positions:
        num = index_to_num.get(lab, lab+1); s = str(num)
        bbox = draw.textbbox((0,0), s, font=font)
        draw.text((x-bbox[2]//2, y-bbox[3]//2), s, fill=0, font=font)
    return canvas.convert("RGB")

def render_legend(palette_map, cols=None):
    rows = len(palette_map)
    if cols is None:
        cols = 2 if rows <= 24 else (3 if rows <= 54 else 4)
    col_w, row_h, pad = 1200, 72, 40
    width = col_w*cols + pad*2
    height = pad*2 + row_h*((rows+cols-1)//cols) + 60
    img = Image.new("RGB", (width, height), (255,255,255)); draw = ImageDraw.Draw(img)
    big, small = load_font(30), load_font(24)
    header = "Farblegende – Amsterdam Standard Series"
    h_bbox = draw.textbbox((0,0), header, font=big)
    draw.text(((width-(h_bbox[2]-h_bbox[0]))//2, 10), header, fill=(0,0,0), font=big)
    per_col, start_y = (rows+cols-1)//cols, 60
    for idx, (num, name, code, rgb) in enumerate(palette_map):
        col, row = idx // per_col, idx % per_col
        x0, y0 = pad + col*col_w, start_y + row*row_h
        draw.text((x0, y0), f"{num:>2}", fill=(0,0,0), font=big)
        sx, sy = x0 + 70, y0 + 12
        draw.rectangle([sx, sy, sx+100, sy+45], fill=rgb, outline=(0,0,0))
        draw.text((sx+120, y0+5), f"{name}", fill=(0,0,0), font=big)
        draw.text((sx+120, y0+38), f"({code})", fill=(90,90,90), font=small)
    return img

def render_preview(arr: np.ndarray,
                   palette_map: List[Tuple[int,str,int,Tuple[int,int,int]]],
                   overlay_boundaries: bool = True,
                   boundary_color: Tuple[int,int,int] = (60, 60, 60),
                   boundary_alpha: int = 120,
                   line_thickness: int = 1):
    label_to_rgb = {i: palette_map[i][3] for i in range(len(palette_map))}
    h, w = arr.shape; preview = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, rgb in label_to_rgb.items():
        preview[arr == lab] = rgb
    img = Image.fromarray(preview)
    if not overlay_boundaries: return img
    boundary_mask = compute_boundaries_no_wrap(Image.fromarray(arr.astype(np.uint8)))
    if line_thickness > 1 and HAVE_SCIPY and maximum_filter is not None:
        boundary_mask = maximum_filter(boundary_mask.astype(np.uint8), size=line_thickness) > 0
    if boundary_alpha <= 0: return img
    a = float(boundary_alpha) / 255.0; base = np.array(img, dtype=np.float32)
    bc = np.array(boundary_color, dtype=np.float32)
    base[boundary_mask] = (1.0 - a) * base[boundary_mask] + a * bc
    return Image.fromarray(base.clip(0,255).astype(np.uint8))

# ==========================================================
# Pipeline
# ==========================================================
def build_pdf(
    input_path: str,
    out_pdf: str,
    colors: int,
    work_w: int,
    work_h: int,
    min_region: int,
    font_size: int,
    thicken_lines: bool,
    merge_close: bool,
    close_thresh: float,
    force_unique: bool,
    autocontrast: bool,
    gamma: float,
    color_boost: float,
    bilateral: bool,
    boundary_color: Tuple[int,int,int],
    boundary_alpha: int,
    line_thickness: int,
    palette: List[Tuple[str,int,Tuple[int,int,int]]],
):
    if not os.path.exists(input_path):
        print(f"❌ Datei nicht gefunden: {input_path}"); return
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}"); return

    progress(5,  "Bild geladen")
    img = center_crop_to_ratio(img, 3/2).resize((work_w, work_h), Image.Resampling.LANCZOS)
    img = preprocess_image(img, do_autocontrast=autocontrast, gamma=gamma,
                           color_boost=color_boost, bilateral=bilateral)
    progress(15, f"Arbeitsauflösung: {work_w}×{work_h}, Farben: {colors}")
    progress(18, "Vorverarbeitung angewendet")

    label_img, pal = quantize_image(img, k=colors)
    arr = np.array(label_img, dtype=np.uint8)
    progress(30, "Quantisierung abgeschlossen")

    arr = merge_small_regions(arr, min_area=min_region)
    progress(45, "Kleine Flächen verschmolzen")

    if force_unique:
        palette_map = assign_unique_amsterdam_weighted(pal, arr, palette)
        progress(60, "Eindeutige Amsterdam-Farben zugeordnet (weighted)")
    else:
        palette_map = map_palette_to_amsterdam(pal, palette)
        arr, palette_map = consolidate_by_amsterdam_code(arr, palette_map,
                                                         merge_close=merge_close,
                                                         close_thresh=close_thresh)
        progress(60, f"Farben konsolidiert (verbleibend: {len(palette_map)})")

    page1 = render_template(arr, palette_map, font_size=font_size, thicken=thicken_lines)
    page2 = render_legend(palette_map, cols=None)
    page3 = render_preview(arr, palette_map, overlay_boundaries=True,
                           boundary_color=boundary_color,
                           boundary_alpha=boundary_alpha,
                           line_thickness=line_thickness)
    progress(85, "Rendering der Seiten")

    dpi = 300
    target_px = (int(60/2.54*dpi), int(40/2.54*dpi))
    page1_big = page1.resize(target_px, Image.Resampling.NEAREST)
    page2_big = page2.resize((target_px[0], int(page2.height*target_px[0]/page2.width)),
                             Image.Resampling.NEAREST)
    page3_big = page3.resize(target_px, Image.Resampling.NEAREST)

    page1_big.save(out_pdf, "PDF", resolution=dpi, save_all=True,
                   append_images=[page2_big, page3_big])
    progress(100, f"OK PDF gespeichert: {out_pdf}")
    print(f"Erstellt am: {datetime.now().isoformat(timespec='seconds')}")

# ==========================================================
# CLI
# ==========================================================
def main():
    ap = argparse.ArgumentParser(description="Malen-nach-Zahlen Generator")
    ap.add_argument("input", help="Eingabebild (JPG/PNG)")
    ap.add_argument("--out", default="malen_nach_zahlen.pdf")
    ap.add_argument("--preset", type=str, help="Preset-Name aus presets.yaml")
    ap.add_argument("--palette", type=str, default="amsterdam_standard.csv", help="Pfad zur CSV-Palette")
    args = ap.parse_args()

    presets = load_presets("presets.yaml")
    if args.preset not in presets:
        print(f"❌ Preset {args.preset} nicht gefunden in presets.yaml"); return
    cfg = presets[args.preset]

    palette = load_palette_csv(args.palette)
    progress(3, f"Palette geladen: {len(palette)} Farben")

    build_pdf(
        input_path=args.input,
        out_pdf=args.out,
        colors=cfg["colors"],
        work_w=cfg["width"],
        work_h=cfg["height"],
        min_region=cfg["detail"]["min_region"],
        font_size=cfg.get("font_size", 9),
        thicken_lines=cfg["detail"].get("thicken_lines", False),
        merge_close=cfg["detail"].get("merge_close", False),
        close_thresh=cfg["detail"].get("close_thresh", 18.0),
        force_unique=cfg.get("force_unique", False),
        autocontrast=cfg["preprocess"].get("autocontrast", True),
        gamma=cfg["preprocess"].get("gamma", 1.0),
        color_boost=cfg["preprocess"].get("colorboost", 1.0),
        bilateral=cfg["preprocess"].get("bilateral", False),
        boundary_color=tuple(cfg.get("boundary_color", (60,60,60))),
        boundary_alpha=cfg.get("boundary_alpha", 120),
        line_thickness=cfg.get("line_thickness", 1),
        palette=palette,
    )

def _print_version():
    print("mnz-generator 1.0.0")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in {"-V", "--version"}:
        _print_version()
    else:
        main()
