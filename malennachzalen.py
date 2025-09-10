import argparse, os, sys, json, csv
from collections import Counter
from typing import List, Tuple, Optional
from datetime import datetime

# --- Drittanbieter ---
try:
    import yaml  # PyYAML
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from skimage import measure
from skimage.color import rgb2lab, deltaE_cie76

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

def progress(step, total_steps, msg, next_msg=None):
    # Prozentsatz für den *Beginn* des Schritts berechnen
    pct = int(round(((step - 1) / total_steps) * 100))
    pct = max(0, min(100, pct))
    bar = _bar(pct)
    
    # Text für den nächsten Schritt formatieren
    next_step_text = f" | Als Nächstes: {next_msg}" if next_msg else ""
    
    # Zeile zusammenbauen und ausgeben
    line = f"{bar} {pct:>3}% [Schritt {step}/{total_steps}] {msg.ljust(35)}{next_step_text}"
    
    # Terminalzeile löschen und neu schreiben
    print(f"\x1b[2K\r{line}", end="")
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
# Presets & Palette laden
# ==========================================================
def _read_csv_dicts(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        buf = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(buf)
        except Exception:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))

def load_palette_csv(path: str) -> List[Tuple[str,int,Tuple[int,int,int]]]:
    rows = []
    for row in _read_csv_dicts(path):
        name = row.get("name") or row.get("Name") or ""
        code = int(row.get("code") or row.get("Code"))
        if "hex" in row and (row["hex"] or row.get("Hex")):
            rgb = hex_to_rgb(row.get("hex") or row.get("Hex"))
        else:
            r = int(row.get("r") or row.get("R"))
            g = int(row.get("g") or row.get("G"))
            b = int(row.get("b") or row.get("B"))
            rgb = (r,g,b)
        rows.append((name, code, rgb))
    if not rows:
        raise ValueError("Palette file is empty")
    return rows

def load_presets(path: str) -> dict:
    if not HAVE_YAML:
        raise ImportError("PyYAML ist nicht installiert. Bitte `pip install pyyaml` ausführen.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("presets.yaml hat kein gültiges Mapping.")
    return data


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
        out = Image.fromarray((arr*255).clip(0,255).astype(np.uint8))
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
def get_color_distance_matrix(colors1: np.ndarray, colors2: np.ndarray, metric="cielab") -> np.ndarray:
    """Berechnet die Distanzmatrix zwischen zwei Paletten."""
    if metric == "cielab":
        # Konvertiere RGB zu LAB. Annahme: RGB ist im Bereich 0-255.
        lab1 = rgb2lab(colors1.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        lab2 = rgb2lab(colors2.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        # Berechne DeltaE 76 für jedes Paar.
        return deltaE_cie76(lab1[:, np.newaxis, :], lab2[np.newaxis, :, :])
    else: # metric == "rgb"
        # Klassische Euklidische Distanz im RGB-Raum.
        return np.sqrt(((colors1[:, None, :] - colors2[None, :, :])**2).sum(axis=2))

def map_palette_to_amsterdam(palette: np.ndarray,
                             amsterdam: List[Tuple[str,int,Tuple[int,int,int]]],
                             metric="cielab"):
    named_rgbs = np.array([rgb for (_,_,rgb) in amsterdam], dtype=np.uint8)
    
    dist_matrix = get_color_distance_matrix(palette, named_rgbs, metric)
    
    mapping = []
    for i in range(palette.shape[0]):
        idx = int(np.argmin(dist_matrix[i, :]))
        name, code, nrgb = amsterdam[idx]
        mapping.append((i+1, name, code, tuple(nrgb)))
    return mapping

def assign_unique_amsterdam_weighted(palette: np.ndarray,
                                     label_img_arr: np.ndarray,
                                     amsterdam: List[Tuple[str,int,Tuple[int,int,int]]],
                                     metric="cielab"):
    src_rgb = palette
    labels, counts = np.unique(label_img_arr, return_counts=True)
    k = src_rgb.shape[0]
    weights = np.ones(k, dtype=np.float32)
    
    # Sicherstellen, dass die Indizes übereinstimmen
    label_map = {label: i for i, label in enumerate(np.unique(label_img_arr))}
    if len(label_map) == k:
        mapped_counts = np.zeros(k)
        for label, count in zip(labels, counts):
            if label in label_map:
                mapped_counts[label_map[label]] = count
        weights = mapped_counts.astype(np.float32)
    else:
        # Fallback, falls die Anzahl der Labels nicht mit der Palettengröße übereinstimmt
        weights[labels] = counts.astype(np.float32)

    weights /= weights.max() + 1e-6

    am_names = [n for (n,_,_) in amsterdam]
    am_codes = [c for (_,c,_) in amsterdam]
    am_rgbs  = np.array([rgb for (*_, rgb) in amsterdam], dtype=np.uint8)

    # Distanzmatrix mit der gewählten Metrik berechnen
    dist_matrix = get_color_distance_matrix(src_rgb, am_rgbs, metric=metric)
    
    # Kostenmatrix mit Gewichtung der Flächengröße
    row_factors = 0.5 + 1.5*weights.reshape(-1,1)
    cost = dist_matrix * row_factors

    # deterministische Reihenfolge
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
    for i in range(src_rgb.shape[0]):
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
    """
    Konsolidiert gleich-zugeordnete Amsterdam-Codes und (optional) mergen ähnlich naher Repräsentanten.
    Gibt (arr_new, final_palette_map) zurück. final_palette_map hat Nummern ab 1.
    """
    # Mappe jeden ursprünglichen Farbindex (1..K) auf einen "neuen Palettenindex"
    code_to_idx = {}
    oldnum_to_newidx = {}
    new_palette = []
    for (num, name, code, rgb) in palette_map:
        if code not in code_to_idx:
            code_to_idx[code] = len(new_palette)
            new_palette.append((name, code, rgb))
        oldnum_to_newidx[num] = code_to_idx[code]

    # Optional: ähnliche Ziel-RGBs clustern
    if merge_close and len(new_palette) > 1:
        reps = np.array([np.array(rgb, float) for (_, _, rgb) in new_palette])
        used = [-1] * len(new_palette)
        rep_meta = []
        for i in range(len(new_palette)):
            if used[i] != -1:
                continue
            rid = len(rep_meta)
            used[i] = rid
            rep_meta.append(new_palette[i])
            d = np.sqrt(((reps - reps[i])**2).sum(1))
            for j in range(i+1, len(new_palette)):
                if used[j] == -1 and d[j] < float(close_thresh):
                    used[j] = rid
        newidx_to_cluster = {i: used[i] for i in range(len(new_palette))}
        oldnum_to_newidx = {old: newidx_to_cluster[new] for old, new in oldnum_to_newidx.items()}
        new_palette = rep_meta

    # LUT anwenden (arr enthält 0..K-1; palette_map-Nummern sind 1..K)
    lut = np.arange(arr.max() + 1)
    for old_num, new_idx in oldnum_to_newidx.items():
        lut[old_num - 1] = new_idx
    arr_new = lut[arr]

    # Auf kompakte 0..N-1 relabeln
    used_vals = np.unique(arr_new)
    remap = {v: i for i, v in enumerate(used_vals)}
    arr_new = np.vectorize(remap.get)(arr_new)

    # finale Palette mit 1..N Nummerierung bauen
    final_palette = []
    for v in used_vals:
        name, code, rgb = new_palette[v]
        final_palette.append((len(final_palette) + 1, name, code, rgb))
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
    color_metric: str,
):
    if not os.path.exists(input_path):
        print(f"❌ Datei nicht gefunden: {input_path}"); return
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}"); return

    # --- Pipeline-Schritte definieren ---
    steps = [
        ("Vorbereitung", lambda: preprocess_image(
            center_crop_to_ratio(img, 3/2).resize((work_w, work_h), Image.Resampling.LANCZOS),
            do_autocontrast=autocontrast, gamma=gamma, color_boost=color_boost, bilateral=bilateral
        )),
        (f"Quantisierung ({colors} Farben)", lambda: quantize_image(img, k=colors)),
        ("Flächenbereinigung", lambda: merge_small_regions(arr, min_area=min_region)),
        ("Farbzuordnung", lambda: (
            assign_unique_amsterdam_weighted(pal, arr, palette) if force_unique
            else consolidate_by_amsterdam_code(
                    arr, map_palette_to_amsterdam(pal, palette),
                    merge_close=merge_close, close_thresh=close_thresh
                 )
        )),
        ("Seiten-Rendering", lambda: (
            render_template(arr, palette_map, font_size=font_size, thicken=thicken_lines),
            render_legend(palette_map, cols=None),
            render_preview(arr, palette_map, overlay_boundaries=True,
                           boundary_color=boundary_color, boundary_alpha=boundary_alpha,
                           line_thickness=line_thickness)
        )),
        ("PDF-Skalierung", lambda: (
            page1.resize((int(60/2.54*300), int(40/2.54*300)), Image.Resampling.NEAREST),
            page2.resize((int(60/2.54*300), int(page2.height*int(60/2.54*300)/page2.width)), Image.Resampling.NEAREST),
            page3.resize((int(60/2.54*300), int(40/2.54*300)), Image.Resampling.NEAREST)
        )),
        ("PDF-Speicherung", lambda: page1_big.save(out_pdf, "PDF", resolution=300, save_all=True, append_images=[page2_big, page3_big]))
    ]
    total_steps = len(steps)

    # --- Pipeline ausführen ---
    for i, (title, func) in enumerate(steps):
        current_step = i + 1
        next_title = steps[i+1][0] if i + 1 < total_steps else "Fertigstellung"
        progress(current_step, total_steps, title, next_msg=next_title)

        if title.startswith("Vorbereitung"):
            img = func()
        elif title.startswith("Quantisierung"):
            label_img, pal = func()
            arr = np.array(label_img, dtype=np.uint8)
        elif title.startswith("Flächenbereinigung"):
            arr = func()
        elif title.startswith("Farbzuordnung"):
            if force_unique:
                palette_map = assign_unique_amsterdam_weighted(pal, arr, palette, metric=color_metric)
            else:
                amsterdam_map = map_palette_to_amsterdam(pal, palette, metric=color_metric)
                arr, palette_map = consolidate_by_amsterdam_code(
                    arr, amsterdam_map,
                    merge_close=merge_close, close_thresh=close_thresh
                )
        elif title.startswith("Seiten-Rendering"):
            page1, page2, page3 = func()
        elif title.startswith("PDF-Skalierung"):
            page1_big, page2_big, page3_big = func()
        elif title.startswith("PDF-Speicherung"):
            func()

    progress(total_steps, total_steps, "PDF gespeichert", "✅")
    print(f"\n\n✅ Fertig! PDF gespeichert unter: {out_pdf}")
    print(f"Erstellt am: {datetime.now().isoformat(timespec='seconds')}")


# ==========================================================
# CLI
# ==========================================================
def main():
    ap = argparse.ArgumentParser(description="Malen-nach-Zahlen Generator")
    ap.add_argument("input", help="Eingabebild (JPG/PNG)")
    ap.add_argument("--out", default="malen_nach_zahlen.pdf", help="Ausgabe-PDF-Datei")
    ap.add_argument("--preset", type=str, help="Preset-Name aus presets.yaml")
    ap.add_argument("--presets", type=str, default="presets.yaml", help="Pfad zur presets.yaml")
    ap.add_argument("--palette", type=str, default="amsterdam_standard.csv", help="Pfad zur CSV-Palette")

    # --- Direkte Konfigurations-Argumente ---
    g = ap.add_argument_group("Allgemeine Einstellungen")
    g.add_argument("--colors", type=int, help="Anzahl der Farben (z.B. 20)")
    g.add_argument("--width", type=int, help="Arbeitsbreite des Bildes (z.B. 1800)")
    g.add_argument("--height", type=int, help="Arbeitshöhe des Bildes (z.B. 1200)")
    g.add_argument("--font-size", type=int, help="Schriftgröße für die Nummern (z.B. 9)")
    g.add_argument("--force-unique", action="store_true", help="Jeder Farbe eine eindeutige Amsterdam-Nr zuweisen")

    g = ap.add_argument_group("Vorverarbeitung")
    g.add_argument("--autocontrast", action=argparse.BooleanOptionalAction, help="Automatischer Kontrast")
    g.add_argument("--gamma", type=float, help="Gamma-Korrektur (z.B. 0.9)")
    g.add_argument("--color-boost", type=float, help="Farb-Verstärkung (z.B. 1.05)")
    g.add_argument("--bilateral", action=argparse.BooleanOptionalAction, help="Bilateral-Filter zur leichten Glättung")

    g = ap.add_argument_group("Detail-Einstellungen")
    g.add_argument("--min-region", type=int, help="Minimale Fläche für eine Region (z.B. 120)")
    g.add_argument("--thicken-lines", action=argparse.BooleanOptionalAction, help="Konturlinien verdicken")
    g.add_argument("--merge-close", action=argparse.BooleanOptionalAction, help="Ähnliche Farben zusammenführen")
    g.add_argument("--close-thresh", type=float, help="Schwellwert für das Zusammenführen (z.B. 18.0)")

    g = ap.add_argument_group("Vorschau-Rendering")
    g.add_argument("--boundary-color", type=str, help="Farbe der Konturlinien (HEX, z.B. #3C3C3C)")
    g.add_argument("--boundary-alpha", type=int, help="Transparenz der Konturlinien (0-255)")
    g.add_argument("--line-thickness", type=int, help="Dicke der Linien in der Vorschau (z.B. 1)")

    ap.add_argument("--color-metric", type=str, default="cielab", choices=["rgb", "cielab"],
                    help="Metrik für den Farbabstand ('rgb' oder 'cielab' für bessere Wahrnehmung)")

    args = ap.parse_args()

    if not HAVE_YAML and args.preset:
        print("❌ PyYAML fehlt. Bitte installieren:  pip install pyyaml")
        sys.exit(1)

    # --- Konfiguration laden & übersteuern ---
    cfg = {
        "colors": 20, "width": 1800, "height": 1200, "font_size": 9,
        "force_unique": False,
        "boundary_color": (60,60,60), "boundary_alpha": 120, "line_thickness": 1,
        "preprocess": {"autocontrast": True, "gamma": 0.9, "colorboost": 1.05, "bilateral": False},
        "detail": {"min_region": 120, "thicken_lines": False, "merge_close": False, "close_thresh": 18.0}
    }

    if args.preset:
        presets = load_presets(args.presets)
        if args.preset not in presets:
            print(f"❌ Preset '{args.preset}' nicht gefunden in {args.presets}")
            print(f"   Verfügbare Presets: {', '.join(sorted(presets.keys()))}")
            sys.exit(1)
        preset_cfg = presets[args.preset]
        cfg.update(preset_cfg)
        if "preprocess" in preset_cfg: cfg["preprocess"].update(preset_cfg["preprocess"])
        if "detail" in preset_cfg: cfg["detail"].update(preset_cfg["detail"])

    # Kommandozeilen-Argumente übersteuern alles
    if args.colors is not None: cfg["colors"] = args.colors
    if args.width is not None: cfg["width"] = args.width
    if args.height is not None: cfg["height"] = args.height
    if args.font_size is not None: cfg["font_size"] = args.font_size
    if args.force_unique: cfg["force_unique"] = True
    if args.boundary_color is not None: cfg["boundary_color"] = hex_to_rgb(args.boundary_color)
    if args.boundary_alpha is not None: cfg["boundary_alpha"] = args.boundary_alpha
    if args.line_thickness is not None: cfg["line_thickness"] = args.line_thickness

    if args.autocontrast is not None: cfg["preprocess"]["autocontrast"] = args.autocontrast
    if args.gamma is not None: cfg["preprocess"]["gamma"] = args.gamma
    if args.color_boost is not None: cfg["preprocess"]["colorboost"] = args.color_boost
    if args.bilateral is not None: cfg["preprocess"]["bilateral"] = args.bilateral

    if args.min_region is not None: cfg["detail"]["min_region"] = args.min_region
    if args.thicken_lines is not None: cfg["detail"]["thicken_lines"] = args.thicken_lines
    if args.merge_close is not None: cfg["detail"]["merge_close"] = args.merge_close
    if args.close_thresh is not None: cfg["detail"]["close_thresh"] = args.close_thresh

    palette = load_palette_csv(args.palette)
    print(f"🎨 Palette '{os.path.basename(args.palette)}' geladen mit {len(palette)} Farben.")

    build_pdf(
        input_path=args.input,
        out_pdf=args.out,
        colors=cfg["colors"],
        work_w=cfg["width"],
        work_h=cfg["height"],
        min_region=cfg["detail"]["min_region"],
        font_size=cfg["font_size"],
        thicken_lines=cfg["detail"]["thicken_lines"],
        merge_close=cfg["detail"]["merge_close"],
        close_thresh=float(cfg["detail"]["close_thresh"]),
        force_unique=cfg["force_unique"],
        autocontrast=bool(cfg["preprocess"]["autocontrast"]),
        gamma=float(cfg["preprocess"]["gamma"]),
        color_boost=float(cfg["preprocess"]["colorboost"]),
        bilateral=bool(cfg["preprocess"]["bilateral"]),
        boundary_color=tuple(cfg["boundary_color"]),
        boundary_alpha=int(cfg["boundary_alpha"]),
        line_thickness=int(cfg["line_thickness"]),
        palette=palette,
        color_metric=args.color_metric,
    )

def _print_version():
    print("mnz-generator 1.0.0")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in {"-V", "--version"}:
        _print_version()
    else:
        main()
