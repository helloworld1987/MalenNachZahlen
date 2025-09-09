import argparse, os, sys, json, csv
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from skimage import measure

# Optional SciPy (Linienverdickung & exakte Zuordnung)
try:
    from scipy.ndimage import maximum_filter
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    linear_sum_assignment = None
    maximum_filter = None

# Optional Bilateral-Filter (sanftes Glätten vor Quantisierung)
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
# Amsterdam-Standardpalette (deine vollständige Liste)
# Format: (Name, Code, (R,G,B))
# ==========================================================
AMSTERDAM_COLORS: List[Tuple[str,int,Tuple[int,int,int]]] = [
    ("Zinkwit", 104, (241,240,236)), ("Titaanwit", 105, (242,241,237)),
    ("Titaanbuff licht", 289, (233,226,213)), ("Titaanbuff donker", 290, (224,213,196)),
    ("Napelsgeel licht", 222, (238,228,198)), ("Napelsgeel groen", 282, (217,215,174)),
    ("Napelsgeel donker", 223, (229,207,158)), ("Napelsgeel rood licht", 292, (227,201,163)),
    ("Napelsgeel Rood", 224, (223,185,143)), ("Permanentcitroengeel licht", 217, (245,230,129)),
    ("Nikkeltitaangeel", 274, (244,223,85)), ("Groengeel", 243, (213,218,61)),
    ("Azogeel Lemon", 267, (242,214,43)), ("Primairgeel", 275, (246,204,36)),
    ("Azogeel licht", 268, (250,204,38)), ("Transparantgeel middel", 272, (240,188,41)),
    ("Azogeel middel", 269, (234,180,44)), ("Azogeel donker", 270, (226,165,45)),
    ("Goudgeel", 253, (219,152,53)), ("Azo-oranje", 276, (221,131,55)),
    ("Vermiljoen", 311, (212,77,56)), ("Naftolrood licht", 398, (211,67,64)),
    ("Naftolrood middel", 396, (205,58,62)), ("Transparantrood middel", 317, (191,60,66)),
    ("Pyrrolerood", 315, (195,50,56)), ("Naftolrood donker", 399, (183,48,61)),
    ("Karmijn", 318, (181,48,64)), ("Primairmagenta", 369, (190,40,80)),
    ("Permanentroodpurper", 348, (179,49,84)), ("Quinacridonerose", 366, (199,67,119)),
    ("Venetiaansrose", 316, (204,126,126)), ("Perzischrose", 330, (200,115,130)),
    ("Lichtrose", 361, (216,134,155)), ("Quinacridonerose licht", 385, (212,112,149)),
    ("Permanent roodviolet licht", 577, (200,82,130)), ("Permanent roodviolet", 567, (174,67,120)),
    ("Caput mortuum violet", 344, (116,69,83)), ("Permanentblauwviolet", 568, (91,63,110)),
    ("Ultramarijnviolet", 507, (78,62,111)), ("Ultramarijnviolet licht", 519, (86,73,123)),
    ("Lila", 556, (167,163,192)), ("Ultramarijn licht", 505, (102,122,179)),
    ("Grijsblauw", 562, (96,122,148)), ("Ultramarijn", 504, (55,87,155)),
    ("Kobaltblauw (ultram.)", 512, (44,88,152)), ("Primaircyaan", 572, (0,118,168)),
    ("Phtaloblauw", 570, (0,81,134)), ("Mangaanblauw phtalo", 582, (0,108,143)),
    ("Koningsblauw", 517, (64,113,162)), ("Briljantblauw", 564, (33,125,176)),
    ("Hemelsblauw licht", 551, (114,163,193)), ("Pruisischblauw (phtalo)", 566, (45,78,97)),
    ("Groenblauw", 557, (0,107,116)), ("Turkooisblauw", 522, (0,124,136)),
    ("Turkooisgroen licht", 660, (97,167,155)), ("Turkooisgroen", 661, (0,129,127)),
    ("Geelgroen licht", 664, (164,185,93)), ("Geelgroen", 617, (151,175,75)),
    ("Briljantgroen", 605, (69,145,80)), ("Permanentgroen licht", 618, (51,132,82)),
    ("Paul veronesegroen", 615, (0,115,96)), ("Permanentgroen donker", 619, (0,92,75)),
    ("Phtalogroen", 575, (0,92,81)), ("Olijfgroen licht", 621, (128,135,80)),
    ("Olijfgroen donker", 622, (91,100,66)), ("Sapgroen", 623, (82,97,56)),
    ("Gele oker", 227, (194,150,64)), ("Goudoker", 231, (192,133,56)),
    ("Sienna naturel", 234, (170,123,75)), ("Sienna gebrand", 411, (149,85,60)),
    ("Omber naturel", 408, (108,93,71)), ("Omber gebrand", 409, (94,77,64)),
    ("Van dijckbruin", 403, (82,71,64)), ("Warmgrijs", 718, (174,165,156)),
    ("Blauwgrijs licht", 750, (147,152,158)), ("Neutraalgrijs", 710, (129,131,132)),
    ("Paynesgrijs", 708, (65,73,80)), ("Lampzwart", 702, (57,57,57)),
    ("Oxydzwart", 735, (58,57,57)), ("Zilver", 800, (171,170,168)),
    ("Lichtgoud", 802, (190,161,109)), ("Donkergoud", 803, (167,126,77)),
    ("Koper", 805, (161,110,81)), ("Brons", 811, (153,124,97)),
    ("Tin", 815, (138,136,134)), ("Grafiet", 840, (112,111,111)),
    ("Metallic geel", 831, (199,180,83)), ("Metallic rood", 832, (179,63,66)),
    ("Metallic violet", 835, (125,86,124)), ("Metallic blauw", 834, (82,111,147)),
    ("Metallic groen", 836, (63,127,107)), ("Metallic zwart", 850, (73,74,74)),
    ("Parel wit", 817, (220,215,207)), ("Parel geel", 818, (216,204,161)),
    ("Parel rood", 819, (195,154,160)), ("Parel blauw", 820, (136,157,185)),
    ("Parel violet", 821, (163,152,178)), ("Parel groen", 822, (165,181,168)),
    ("Reflex geel", 256, (227,224,46)), ("Reflex oranje", 257, (240,90,50)),
    ("Reflex rose", 384, (234,55,100)), ("Reflex groen", 672, (157,207,64)),
]

# ==========================================================
# Externe Palette (optional)
# ==========================================================
def load_palette_file(path: str) -> List[Tuple[str,int,Tuple[int,int,int]]]:
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext in (".csv", ".tsv"):
        with open(path, "r", encoding="utf-8-sig") as f:
            dialect = csv.Sniffer().sniff(f.read(4096))
            f.seek(0); reader = csv.DictReader(f, dialect=dialect)
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
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                name = item["name"]; code = int(item["code"])
                rgb = hex_to_rgb(item["hex"]) if "hex" in item else tuple(map(int, item["rgb"]))
                rows.append((name, code, rgb))
    else:
        raise ValueError("Unsupported palette file. Use .csv/.tsv or .json")
    if not rows: raise ValueError("Palette file is empty")
    return rows

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
# Konturen (ohne Wrap)
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

def consolidate_by_amsterdam_code(arr: np.ndarray,
                                  palette_map: List[Tuple[int,str,int,Tuple[int,int,int]]],
                                  merge_close=False, close_thresh=18.0):
    code_to_idx = {}; oldnum_to_newidx = {}; new_palette = []
    for (num, name, code, rgb) in palette_map:
        if code not in code_to_idx:
            code_to_idx[code] = len(new_palette); new_palette.append((name, code, rgb))
        oldnum_to_newidx[num] = code_to_idx[code]

    if merge_close and len(new_palette) > 1:
        reps = np.array([np.array(rgb, float) for (_, _, rgb) in new_palette])
        used = [-1] * len(new_palette); rep_meta = []
        for i in range(len(new_palette)):
            if used[i] != -1: continue
            rid = len(rep_meta); used[i] = rid; rep_meta.append(new_palette[i])
            d = np.sqrt(((reps - reps[i])**2).sum(1))
            for j in range(i+1, len(new_palette)):
                if used[j] == -1 and d[j] < close_thresh: used[j] = rid
        newidx_to_cluster = {i: used[i] for i in range(len(new_palette))}
        oldnum_to_newidx = {old: newidx_to_cluster[new] for old, new in oldnum_to_newidx.items()}
        new_palette = rep_meta

    lut = np.arange(arr.max()+1)
    for old_num, new_idx in oldnum_to_newidx.items():
        lut[old_num-1] = new_idx
    arr_new = lut[arr]

    used_vals = np.unique(arr_new); remap = {v: i for i, v in enumerate(used_vals)}
    arr_new = np.vectorize(remap.get)(arr_new)

    final_palette = []
    for v in used_vals:
        name, code, rgb = new_palette[v]
        final_palette.append((len(final_palette)+1, name, code, rgb))
    return arr_new, final_palette

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
# Rendering (Vorlage, Legende, Vorschau)
# ==========================================================
def render_template(arr, palette_map, font_size=8, thicken=False):
    boundary = compute_boundaries_no_wrap(Image.fromarray(arr.astype(np.uint8), "L"))
    if thicken and HAVE_SCIPY and maximum_filter is not None:
        boundary = maximum_filter(boundary.astype(np.uint8), size=2) > 0
    canvas = Image.new("L", arr.shape[::-1], 255)
    boundary_img = Image.fromarray(np.where(boundary, 0, 255).astype(np.uint8), "L")
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

def render_legend(palette_map, cols=2):
    rows = len(palette_map); col_w, row_h, pad = 1200, 72, 40
    width = col_w*cols + pad*2; height = pad*2 + row_h*((rows+cols-1)//cols) + 60
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
    """Seite 3: farbige Vorschau – Flächen ausmalen, Konturen oben drauf."""
    label_to_rgb = {i: palette_map[i][3] for i in range(len(palette_map))}
    h, w = arr.shape; preview = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, rgb in label_to_rgb.items():
        preview[arr == lab] = rgb
    img = Image.fromarray(preview, "RGB")
    if not overlay_boundaries: return img
    boundary_mask = compute_boundaries_no_wrap(Image.fromarray(arr.astype(np.uint8), "L"))
    if line_thickness > 1 and HAVE_SCIPY and maximum_filter is not None:
        boundary_mask = maximum_filter(boundary_mask.astype(np.uint8), size=line_thickness) > 0
    if boundary_alpha <= 0: return img
    a = float(boundary_alpha) / 255.0; base = np.array(img, dtype=np.float32)
    bc = np.array(boundary_color, dtype=np.float32)
    base[boundary_mask] = (1.0 - a) * base[boundary_mask] + a * bc
    return Image.fromarray(base.clip(0,255).astype(np.uint8), "RGB")

# ==========================================================
# Presets
# ==========================================================
PRESETS = {
    "sea-portrait": dict(
        colors=72, width=6000, height=4000, min_region=80, font_size=10,
        thicken_lines=False, merge_close=False, close_thresh=20.0,
        force_unique=False,
        autocontrast=True, gamma=1.00, color_boost=1.05, bilateral=False,
        boundary_color=(60,60,60), boundary_alpha=120, line_thickness=1
    ),
    "portrait": dict(
        colors=24, width=2000, height=1334, min_region=90, font_size=11,
        thicken_lines=False, merge_close=False, close_thresh=18.0,
        force_unique=False,
        autocontrast=True, gamma=0.90, color_boost=1.08, bilateral=True,
        boundary_color=(60,60,60), boundary_alpha=130, line_thickness=1
    ),
    "landscape-sky": dict(
        colors=26, width=2400, height=1600, min_region=120, font_size=9,
        thicken_lines=False, merge_close=False, close_thresh=20.0,
        force_unique=False,
        autocontrast=True, gamma=0.95, color_boost=1.03, bilateral=True,
        boundary_color=(70,70,70), boundary_alpha=120, line_thickness=1
    ),
    "high-contrast": dict(
        colors=18, width=2000, height=1334, min_region=140, font_size=9,
        thicken_lines=True, merge_close=True, close_thresh=16.0,
        force_unique=False,
        autocontrast=True, gamma=0.85, color_boost=1.10, bilateral=False,
        boundary_color=(40,40,40), boundary_alpha=160, line_thickness=2
    ),
    "draft-fast": dict(
        colors=16, width=1600, height=1066, min_region=120, font_size=8,
        thicken_lines=False, merge_close=False, close_thresh=18.0,
        force_unique=False,
        autocontrast=True, gamma=1.00, color_boost=1.00, bilateral=False,
        boundary_color=(70,70,70), boundary_alpha=150, line_thickness=1
    ),
}

def apply_preset(args, preset_name):
    cfg = PRESETS[preset_name].copy()
    # Nur Parameter überschreiben, die der Nutzer EXPLIZIT setzt (args.* != None)
    def pick(name, val): return val if val is not None else cfg.get(name)
    merged = dict(
        colors=pick("colors", args.colors),
        width=pick("width", args.width),
        height=pick("height", args.height),
        min_region=pick("min_region", args.min_region),
        font_size=pick("font_size", args.font_size),
        thicken_lines=pick("thicken_lines", args.thicken_lines),
        merge_close=pick("merge_close", args.merge_close),
        close_thresh=pick("close_thresh", args.close_thresh),
        force_unique=pick("force_unique", args.force_unique),
        autocontrast=pick("autocontrast", args.autocontrast),
        gamma=pick("gamma", args.gamma),
        color_boost=pick("color_boost", args.color_boost),
        bilateral=pick("bilateral", args.bilateral),
        boundary_color=pick("boundary_color", tuple(args.boundary_color) if args.boundary_color else None),
        boundary_alpha=pick("boundary_alpha", args.boundary_alpha),
        line_thickness=pick("line_thickness", args.line_thickness),
    )
    return merged

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
    palette_path: Optional[str] = None,
):
    if not os.path.exists(input_path):
        print(f"❌ Datei nicht gefunden: {input_path}"); return
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}"); return

    # Palette (intern oder extern)
    if palette_path:
        try:
            amsterdam = load_palette_file(palette_path)
            progress(3, f"Palette geladen: {len(amsterdam)} Farben")
        except Exception as e:
            print(f"⚠️  Konnte Palette nicht laden ({e}). Nutze interne Liste.")
            amsterdam = AMSTERDAM_COLORS
    else:
        amsterdam = AMSTERDAM_COLORS
        progress(3, f"Interne Palette aktiv: {len(amsterdam)} Farben")

    progress(5,  "Bild geladen")
    img = center_crop_to_ratio(img, 3/2).resize((work_w, work_h), Image.Resampling.LANCZOS)
    img = preprocess_image(img, do_autocontrast=autocontrast, gamma=gamma,
                           color_boost=color_boost, bilateral=bilateral)
    progress(18, "Vorverarbeitung angewendet")

    label_img, pal = quantize_image(img, k=colors)
    arr = np.array(label_img, dtype=np.uint8)
    progress(30, "Quantisierung abgeschlossen")

    arr = merge_small_regions(arr, min_area=min_region)
    progress(45, "Kleine Flächen verschmolzen")

    if force_unique:
        palette_map = assign_unique_amsterdam_weighted(pal, arr, amsterdam)
        progress(60, "Eindeutige Amsterdam-Farben zugeordnet (weighted)")
    else:
        palette_map = map_palette_to_amsterdam(pal, amsterdam)
        arr, palette_map = consolidate_by_amsterdam_code(arr, palette_map,
                                                         merge_close=merge_close,
                                                         close_thresh=close_thresh)
        progress(60, f"Farben konsolidiert (verbleibend: {len(palette_map)})")

    page1 = render_template(arr, palette_map, font_size=font_size, thicken=thicken_lines)
    legend_cols = 2 if len(palette_map) <= 24 else 3
    page2 = render_legend(palette_map, cols=legend_cols)
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
    progress(100, f"✅ PDF gespeichert: {out_pdf}")

# ==========================================================
# CLI
# ==========================================================
def main():
    ap = argparse.ArgumentParser(description="Malen-nach-Zahlen Generator (Presets, Amsterdam, Vorschau, Fortschritt)")
    ap.add_argument("input", help="Eingabebild (JPG/PNG)")
    ap.add_argument("--out", default="malen_nach_zahlen.pdf")

    # Defaults = None -> so erkennen wir, was der Nutzer explizit setzt (→ Preset kann Rest füllen)
    ap.add_argument("--preset", choices=list(PRESETS.keys()))
    ap.add_argument("--colors", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--min-region", type=int, default=None)
    ap.add_argument("--font-size", type=int, default=None)
    ap.add_argument("--thicken-lines", action="store_true", default=None)
    ap.add_argument("--merge-close", action="store_true", default=None)
    ap.add_argument("--close-thresh", type=float, default=None)
    ap.add_argument("--force-unique", action="store_true", default=None)

    # Preprocess
    ap.add_argument("--autocontrast", action="store_true", default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--color-boost", type=float, default=None)
    ap.add_argument("--bilateral", action="store_true", default=None)

    # Preview line
    ap.add_argument("--boundary-color", nargs=3, type=int, default=None, metavar=("R","G","B"))
    ap.add_argument("--boundary-alpha", type=int, default=None)
    ap.add_argument("--line-thickness", type=int, default=None)

    ap.add_argument("--palette", type=str, default=None,
                    help="Externe Palette (CSV/JSON) mit name,code,hex ODER name,code,r,g,b")

    args = ap.parse_args()

    # Basis-Defaults (falls kein Preset & keine Flags)
    base = dict(colors=20, width=1800, height=1200, min_region=120, font_size=8,
                thicken_lines=False, merge_close=False, close_thresh=18.0,
                force_unique=False, autocontrast=True, gamma=0.90, color_boost=1.05,
                bilateral=False, boundary_color=(60,60,60), boundary_alpha=120, line_thickness=1)

    cfg = base if args.preset is None else apply_preset(args, args.preset)

    # Falls kein Preset, aber Nutzer hat Flags gesetzt → über base drübermappen
    if args.preset is None:
        def pick(name, val): return cfg[name] if val is None else (tuple(val) if name=="boundary_color" and val is not None else val)
        cfg = dict(
            colors=pick("colors", args.colors),
            width=pick("width", args.width),
            height=pick("height", args.height),
            min_region=pick("min_region", args.min_region),
            font_size=pick("font_size", args.font_size),
            thicken_lines=pick("thicken_lines", args.thicken_lines),
            merge_close=pick("merge_close", args.merge_close),
            close_thresh=pick("close_thresh", args.close_thresh),
            force_unique=pick("force_unique", args.force_unique),
            autocontrast=pick("autocontrast", args.autocontrast),
            gamma=pick("gamma", args.gamma),
            color_boost=pick("color_boost", args.color_boost),
            bilateral=pick("bilateral", args.bilateral),
            boundary_color=pick("boundary_color", args.boundary_color),
            boundary_alpha=pick("boundary_alpha", args.boundary_alpha),
            line_thickness=pick("line_thickness", args.line_thickness),
        )

    build_pdf(
        input_path=args.input,
        out_pdf=args.out,
        colors=cfg["colors"],
        work_w=cfg["width"],
        work_h=cfg["height"],
        min_region=cfg["min_region"],
        font_size=cfg["font_size"],
        thicken_lines=cfg["thicken_lines"],
        merge_close=cfg["merge_close"],
        close_thresh=cfg["close_thresh"],
        force_unique=cfg["force_unique"],
        autocontrast=cfg["autocontrast"],
        gamma=cfg["gamma"],
        color_boost=cfg["color_boost"],
        bilateral=cfg["bilateral"],
        boundary_color=cfg["boundary_color"],
        boundary_alpha=cfg["boundary_alpha"],
        line_thickness=cfg["line_thickness"],
        palette_path=args.palette,
    )

if __name__ == "__main__":
    main()
