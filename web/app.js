// Main Application Logic for Malen nach Zahlen Studio

let worker = null;
let originalImage = null;
let processedData = null;
let currentView = 'template'; // 'template', 'preview', 'split', 'original'
let activeHighlightNumber = null;

// Zoom & Pan state
let scale = 1.0;
let panX = 0;
let panY = 0;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;

// Split Slider State
let splitPos = 0.5; // 0..1
let isDraggingSplit = false;

// DOM Elements
const canvasStage = document.getElementById('canvasStage');
const canvasWrapper = document.getElementById('canvasTransformWrapper');
const mainCanvas = document.getElementById('mainCanvas');
const splitCanvas = document.getElementById('splitOverlayCanvas');
const splitDivider = document.getElementById('splitDivider');
const mainCtx = mainCanvas.getContext('2d');
const splitCtx = splitCanvas.getContext('2d');

const fileInput = document.getElementById('fileInput');
const btnSample = document.getElementById('btnSample');
const btnEmptySample = document.getElementById('btnEmptySample');
const btnProcess = document.getElementById('btnProcess');
const emptyState = document.getElementById('emptyState');

// Controls
const sliderColors = document.getElementById('sliderColors');
const valColors = document.getElementById('valColors');
const sliderMinRegion = document.getElementById('sliderMinRegion');
const valMinRegion = document.getElementById('valMinRegion');
const checkSmoothing = document.getElementById('checkSmoothing');

const sliderGamma = document.getElementById('sliderGamma');
const valGamma = document.getElementById('valGamma');
const sliderContrast = document.getElementById('sliderContrast');
const valContrast = document.getElementById('valContrast');
const sliderSaturation = document.getElementById('sliderSaturation');
const valSaturation = document.getElementById('valSaturation');

const sliderLineWidth = document.getElementById('sliderLineWidth');
const valLineWidth = document.getElementById('valLineWidth');
const sliderNumberScale = document.getElementById('sliderNumberScale');
const valNumberScale = document.getElementById('valNumberScale');
const selectPaperFormat = document.getElementById('selectPaperFormat');
const checkAcrylicEffect = document.getElementById('checkAcrylicEffect');

// Progress
const progressBox = document.getElementById('progressBox');
const progressText = document.getElementById('progressText');
const progressBarFill = document.getElementById('progressBarFill');

// Palette
const paletteList = document.getElementById('paletteList');
const paletteCountBadge = document.getElementById('paletteCountBadge');
const highlightBanner = document.getElementById('highlightBanner');
const highlightBannerText = document.getElementById('highlightBannerText');
const btnClearHighlight = document.getElementById('btnClearHighlight');

// Tooltip
const canvasTooltip = document.getElementById('canvasTooltip');
const tooltipSwatch = document.getElementById('tooltipSwatch');
const tooltipTitle = document.getElementById('tooltipTitle');
const tooltipSub = document.getElementById('tooltipSub');

// Zoom DOM
const btnZoomIn = document.getElementById('btnZoomIn');
const btnZoomOut = document.getElementById('btnZoomOut');
const btnZoomFit = document.getElementById('btnZoomFit');
const btnZoomReset = document.getElementById('btnZoomReset');
const zoomLevelText = document.getElementById('zoomLevelText');

let uncroppedImage = null;
let isCropped = false;
const btnCrop = document.getElementById('btnCrop');

// Presets
const PRESETS = {
  street: {
    colors: 24, minRegion: 450, smooth: true,
    gamma: 0.95, contrast: 1.10, saturation: 1.25, lineWidth: 1.0
  },
  portrait: {
    colors: 24, minRegion: 120, smooth: true,
    gamma: 0.90, contrast: 1.05, saturation: 1.08, lineWidth: 1.0
  },
  landscape: {
    colors: 28, minRegion: 160, smooth: true,
    gamma: 1.00, contrast: 1.00, saturation: 1.05, lineWidth: 1.0
  },
  popart: {
    colors: 16, minRegion: 200, smooth: false,
    gamma: 0.85, contrast: 1.25, saturation: 1.30, lineWidth: 1.5
  },
  beginner: {
    colors: 14, minRegion: 350, smooth: true,
    gamma: 1.00, contrast: 1.00, saturation: 1.00, lineWidth: 1.2
  },
  detailed: {
    colors: 36, minRegion: 60, smooth: false,
    gamma: 1.00, contrast: 1.00, saturation: 1.05, lineWidth: 1.0
  }
};

// Initialize
function init() {
  initWorker();
  initEvents();
  setupCollapsibles();

  // Load sample image on start
  loadSampleImage();
}

function initWorker() {
  try {
    worker = new Worker('worker.js');
  } catch (err) {
    console.error('Worker initialization failed:', err);
  }

  worker.onmessage = function (e) {
    const { type, step, total, msg, result, error } = e.data;

    if (type === 'progress') {
      progressBox.style.display = 'block';
      progressText.textContent = msg;
      const pct = Math.round((step / total) * 100);
      progressBarFill.style.width = `${pct}%`;
    } else if (type === 'complete') {
      progressBox.style.display = 'none';
      processedData = result;
      btnProcess.disabled = false;
      renderCurrentView();
      buildPaletteUI();
    } else if (type === 'error') {
      progressBox.style.display = 'none';
      btnProcess.disabled = false;
      alert('Fehler bei der Berechnung: ' + error);
      console.error(error);
    }
  };
}

function initEvents() {
  // File input
  fileInput.addEventListener('change', handleFileSelect);
  btnSample.addEventListener('click', loadSampleImage);
  if (btnEmptySample) btnEmptySample.addEventListener('click', loadSampleImage);
  btnProcess.addEventListener('click', runPipeline);

  if (btnCrop) btnCrop.addEventListener('click', toggleCrop);

  // Sliders
  sliderColors.addEventListener('input', () => valColors.textContent = sliderColors.value);
  sliderMinRegion.addEventListener('input', () => valMinRegion.textContent = `${sliderMinRegion.value} px`);
  sliderGamma.addEventListener('input', () => valGamma.textContent = parseFloat(sliderGamma.value).toFixed(2));
  sliderContrast.addEventListener('input', () => valContrast.textContent = parseFloat(sliderContrast.value).toFixed(2));
  sliderSaturation.addEventListener('input', () => valSaturation.textContent = parseFloat(sliderSaturation.value).toFixed(2));
  sliderLineWidth.addEventListener('input', () => {
    valLineWidth.textContent = `${sliderLineWidth.value} px`;
    if (processedData) renderCurrentView();
  });
  sliderNumberScale.addEventListener('input', () => {
    valNumberScale.textContent = `${sliderNumberScale.value}%`;
    if (processedData) renderCurrentView();
  });
  if (checkAcrylicEffect) {
    checkAcrylicEffect.addEventListener('change', () => {
      if (processedData) renderCurrentView();
    });
  }

  // Presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyPreset(btn.dataset.preset);
    });
  });

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentView = btn.dataset.view;
      renderCurrentView();
    });
  });

  // Canvas Pan & Zoom
  canvasStage.addEventListener('wheel', handleWheel, { passive: false });
  canvasStage.addEventListener('mousedown', handleMouseDown);
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);

  // Split Divider Drag
  splitDivider.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    isDraggingSplit = true;
  });

  // Zoom buttons
  btnZoomIn.addEventListener('click', () => setZoom(scale * 1.25));
  btnZoomOut.addEventListener('click', () => setZoom(scale / 1.25));
  btnZoomFit.addEventListener('click', zoomToFit);
  btnZoomReset.addEventListener('click', () => setZoom(1.0));

  // Clear highlight
  btnClearHighlight.addEventListener('click', clearHighlight);

  // Dropdown
  const btnExportMenu = document.getElementById('btnExportMenu');
  const exportDropdown = document.getElementById('exportDropdown');
  btnExportMenu.addEventListener('click', (e) => {
    e.stopPropagation();
    btnExportMenu.parentElement.classList.toggle('open');
  });
  window.addEventListener('click', () => {
    document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
  });

  // Export buttons
  document.getElementById('btnExportPdf').addEventListener('click', exportPdf);
  document.getElementById('btnExportSvg').addEventListener('click', exportSvg);
  document.getElementById('btnExportPng').addEventListener('click', exportPng);
  document.getElementById('btnExportPreviewPng').addEventListener('click', exportPreviewPng);
  document.getElementById('btnPrint').addEventListener('click', () => window.print());

  // Drag and Drop files onto stage
  canvasStage.addEventListener('dragover', (e) => e.preventDefault());
  canvasStage.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadImageFromFile(e.dataTransfer.files[0]);
    }
  });

  // Hover over canvas to detect region
  canvasStage.addEventListener('mousemove', handleCanvasHover);
  canvasStage.addEventListener('mouseleave', () => canvasTooltip.style.display = 'none');
  canvasStage.addEventListener('click', handleCanvasClick);
}

function setupCollapsibles() {
  ['headerPhotoAdjust', 'headerViewSettings'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('click', () => {
        el.parentElement.classList.toggle('collapsed');
      });
    }
  });
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;

  sliderColors.value = p.colors;
  valColors.textContent = p.colors;

  sliderMinRegion.value = p.minRegion;
  valMinRegion.textContent = `${p.minRegion} px`;

  checkSmoothing.checked = p.smooth;

  sliderGamma.value = p.gamma;
  valGamma.textContent = p.gamma.toFixed(2);

  sliderContrast.value = p.contrast;
  valContrast.textContent = p.contrast.toFixed(2);

  sliderSaturation.value = p.saturation;
  valSaturation.textContent = p.saturation.toFixed(2);

  if (p.lineWidth) {
    sliderLineWidth.value = p.lineWidth;
    valLineWidth.textContent = `${p.lineWidth} px`;
  }

  // Trigger processing
  if (originalImage) {
    runPipeline();
  }
}

// Image Loading & Crop
function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    loadImageFromFile(e.target.files[0]);
  }
}

function loadImageFromFile(file) {
  const reader = new FileReader();
  reader.onload = function (evt) {
    const img = new Image();
    img.onload = function () {
      uncroppedImage = img;
      originalImage = img;
      isCropped = false;
      if (btnCrop) btnCrop.textContent = '✂️ Zuschneiden';
      emptyState.style.display = 'none';
      zoomToFit();
      runPipeline();
    };
    img.src = evt.target.result;
  };
  reader.readAsDataURL(file);
}

function loadSampleImage() {
  const img = new Image();
  img.onload = function () {
    uncroppedImage = img;
    originalImage = img;
    isCropped = false;
    if (btnCrop) btnCrop.textContent = '✂️ Zuschneiden';
    emptyState.style.display = 'none';
    zoomToFit();
    runPipeline();
  };
  img.src = 'sample.png';
}

function toggleCrop() {
  if (!uncroppedImage) return;

  if (!isCropped) {
    // Zoom in on the main subject (children walking down sidewalk)
    const ow = uncroppedImage.naturalWidth || uncroppedImage.width;
    const oh = uncroppedImage.naturalHeight || uncroppedImage.height;

    // Focus crop: 55% width, 68% height centered on sidewalk & kids
    const sw = Math.round(ow * 0.56);
    const sh = Math.round(oh * 0.68);
    const sx = Math.round((ow - sw) * 0.58);
    const sy = Math.round((oh - sh) * 0.38);

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = sw;
    cropCanvas.height = sh;
    const ctx = cropCanvas.getContext('2d');
    ctx.drawImage(uncroppedImage, sx, sy, sw, sh, 0, 0, sw, sh);

    const croppedImg = new Image();
    croppedImg.onload = function () {
      originalImage = croppedImg;
      isCropped = true;
      if (btnCrop) btnCrop.textContent = '↩️ Vollbild';
      zoomToFit();
      runPipeline();
    };
    croppedImg.src = cropCanvas.toDataURL('image/jpeg', 0.95);
  } else {
    originalImage = uncroppedImage;
    isCropped = false;
    if (btnCrop) btnCrop.textContent = '✂️ Zuschneiden';
    zoomToFit();
    runPipeline();
  }
}

// Processing Pipeline
function runPipeline() {
  if (!originalImage) return;

  btnProcess.disabled = true;
  progressBox.style.display = 'block';
  progressBarFill.style.width = '5%';
  progressText.textContent = 'Berechnung wird gestartet...';

  // Scale down oversized images for smooth interactive processing
  // Max dimension 1400px (ideal balance between fine detail and instant response)
  const maxDim = 1400;
  let w = originalImage.naturalWidth || originalImage.width;
  let h = originalImage.naturalHeight || originalImage.height;

  if (w > maxDim || h > maxDim) {
    const r = Math.min(maxDim / w, maxDim / h);
    w = Math.round(w * r);
    h = Math.round(h * r);
  }

  // Draw to offscreen canvas to extract raw ImageData
  const offCanvas = document.createElement('canvas');
  offCanvas.width = w;
  offCanvas.height = h;
  const offCtx = offCanvas.getContext('2d');
  offCtx.drawImage(originalImage, 0, 0, w, h);
  const imageData = offCtx.getImageData(0, 0, w, h);

  const config = {
    colors: parseInt(sliderColors.value, 10),
    minRegionSize: parseInt(sliderMinRegion.value, 10),
    smooth: checkSmoothing.checked,
    gamma: parseFloat(sliderGamma.value),
    contrast: parseFloat(sliderContrast.value),
    saturation: parseFloat(sliderSaturation.value)
  };

  worker.postMessage({
    type: 'process',
    imageData,
    config
  });
}

// Canvas Rendering
function renderCurrentView() {
  if (!processedData || !originalImage) return;

  const { width, height } = processedData;

  if (mainCanvas.width !== width || mainCanvas.height !== height) {
    mainCanvas.width = width;
    mainCanvas.height = height;
    splitCanvas.width = width;
    splitCanvas.height = height;
  }

  // Visibility of split divider
  if (currentView === 'split') {
    splitCanvas.style.display = 'block';
    splitDivider.style.display = 'block';
    updateSplitPosition();
  } else {
    splitCanvas.style.display = 'none';
    splitDivider.style.display = 'none';
  }

  if (currentView === 'template') {
    renderTemplate(mainCtx, activeHighlightNumber);
  } else if (currentView === 'preview') {
    renderPreview(mainCtx, activeHighlightNumber);
  } else if (currentView === 'original') {
    renderOriginal(mainCtx);
  } else if (currentView === 'split') {
    // Left side: original, Right side: preview or template
    renderOriginal(mainCtx);
    renderPreview(splitCtx, activeHighlightNumber);
    clipSplitCanvas();
  }
}

function renderTemplate(ctx, highlightNum = null) {
  const { width, height, boundaries, regions, palette } = processedData;
  const lineWidth = parseFloat(sliderLineWidth.value);
  const numScale = parseInt(sliderNumberScale.value, 10) / 100.0;

  // 1. Fill background white
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // 2. If a color is highlighted, paint its regions in light accent
  if (highlightNum !== null) {
    const colObj = palette.find(p => p.number === highlightNum);
    const hex = colObj ? colObj.hex : '#fef08a';

    // Highlight fill
    ctx.fillStyle = hex + '55'; // translucent
    const labels = processedData.labels;
    // Fast pixel fill for highlighted regions
    const imgData = ctx.getImageData(0, 0, width, height);
    const d = imgData.data;

    const rgb = colObj ? colObj.rgb : [254, 240, 138];

    for (let i = 0; i < labels.length; i++) {
      const reg = regions[labels[i]];
      if (reg && reg.number === highlightNum) {
        const p = i * 4;
        d[p] = Math.round(d[p] * 0.4 + rgb[0] * 0.6);
        d[p + 1] = Math.round(d[p + 1] * 0.4 + rgb[1] * 0.6);
        d[p + 2] = Math.round(d[p + 2] * 0.4 + rgb[2] * 0.6);
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // 3. Draw Boundaries
  const imgData = ctx.getImageData(0, 0, width, height);
  const d = imgData.data;
  const boundaryColor = [55, 60, 68]; // Dark slate

  for (let i = 0; i < boundaries.length; i++) {
    if (boundaries[i] === 1) {
      const p = i * 4;
      d[p] = boundaryColor[0];
      d[p + 1] = boundaryColor[1];
      d[p + 2] = boundaryColor[2];
    }
  }
  ctx.putImageData(imgData, 0, 0);

  // 4. Draw Numbers at Polylabel Centers
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < regions.length; i++) {
    const reg = regions[i];
    if (reg.radius < 3) continue; // Skip tiny regions to avoid clutter

    const isTarget = highlightNum !== null && reg.number === highlightNum;
    const baseFontSize = Math.max(7, Math.min(18, Math.round(reg.radius * 1.35)));
    const finalSize = Math.round(baseFontSize * numScale);

    ctx.font = `600 ${finalSize}px -apple-system, sans-serif`;

    if (isTarget) {
      // Glow pill behind number
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(reg.x, reg.y, finalSize * 0.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
    } else if (highlightNum !== null) {
      ctx.fillStyle = '#94a3b8'; // Muted
    } else {
      ctx.fillStyle = '#1e293b'; // Normal dark
    }

    ctx.fillText(String(reg.number), reg.x, reg.y + 0.5);
  }
}

function renderPreview(ctx, highlightNum = null) {
  const { width, height, labels, regions, palette, boundaries } = processedData;

  const imgData = ctx.createImageData(width, height);
  const d = imgData.data;

  // Build color map: regionId -> [r, g, b]
  const colorMap = new Uint8ClampedArray(regions.length * 3);
  for (let r = 0; r < regions.length; r++) {
    const reg = regions[r];
    const col = palette.find(p => p.number === reg.number);
    if (col) {
      let rgb = col.rgb;
      if (highlightNum !== null && reg.number !== highlightNum) {
        // Desaturate non-highlighted colors
        const gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        rgb = [
          Math.round(gray * 0.7 + rgb[0] * 0.3),
          Math.round(gray * 0.7 + rgb[1] * 0.3),
          Math.round(gray * 0.7 + rgb[2] * 0.3)
        ];
      }
      colorMap[r * 3] = rgb[0];
      colorMap[r * 3 + 1] = rgb[1];
      colorMap[r * 3 + 2] = rgb[2];
    }
  }

  for (let i = 0; i < labels.length; i++) {
    const regId = labels[i];
    const cOffset = regId * 3;
    const p = i * 4;

    if (boundaries[i] === 1) {
      // Subtle boundary overlay
      d[p] = Math.round(colorMap[cOffset] * 0.65);
      d[p + 1] = Math.round(colorMap[cOffset + 1] * 0.65);
      d[p + 2] = Math.round(colorMap[cOffset + 2] * 0.65);
    } else {
      d[p] = colorMap[cOffset];
      d[p + 1] = colorMap[cOffset + 1];
      d[p + 2] = colorMap[cOffset + 2];
    }
    d[p + 3] = 255;
  }

  // Optional Acrylic & Canvas Texture (Impasto, Pinselduktus & Leinwandgewebe)
  const isAcrylic = !checkAcrylicEffect || checkAcrylicEffect.checked;
  if (isAcrylic && highlightNum === null) {
    applyAcrylicCanvasTexture(d, width, height);
  }

  ctx.putImageData(imgData, 0, 0);

  // If highlighted, outline target regions prominently
  if (highlightNum !== null) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = 'bold 12px sans-serif';

    for (let i = 0; i < regions.length; i++) {
      const reg = regions[i];
      if (reg.number === highlightNum && reg.radius >= 4) {
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = '#000000';
        ctx.shadowBlur = 4;
        ctx.fillText(String(reg.number), reg.x, reg.y);
      }
    }
    ctx.shadowBlur = 0;
  }
}

// Procedural Acrylic Impasto & Canvas Weave Filter
function applyAcrylicCanvasTexture(data, width, height) {
  // Precompute luminance map for 3D directional lighting
  const lum = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const p = i * 4;
    lum[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }

  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOffset + x;
      const p = idx * 4;

      // 1. Linen Canvas Weave Texture (micro-fabric weave)
      const weaveX = Math.sin(x * 1.57);
      const weaveY = Math.sin(y * 1.57);
      const canvasWeave = (weaveX * weaveY) * 5.5;

      // 2. 3D Paint Ridge & Impasto (Light source from top-left, 45 degrees)
      const dx = lum[idx] - lum[idx - 1];
      const dy = lum[idx] - lum[idx - width];
      const slope = (dx + dy) * 0.38;

      // 3. Directional Brush Stroke Texture
      const strokeGrain = Math.sin((x * 0.85 + y * 0.45) * 0.6) * 3.5;

      const totalEffect = canvasWeave + slope + strokeGrain;

      data[p] = Math.max(0, Math.min(255, data[p] + totalEffect));
      data[p + 1] = Math.max(0, Math.min(255, data[p + 1] + totalEffect));
      data[p + 2] = Math.max(0, Math.min(255, data[p + 2] + totalEffect));
    }
  }
}

function renderOriginal(ctx) {
  if (!originalImage || !processedData) return;
  ctx.drawImage(originalImage, 0, 0, processedData.width, processedData.height);
}

function clipSplitCanvas() {
  const { width, height } = processedData;
  const splitX = Math.round(width * splitPos);

  splitCanvas.style.clipPath = `polygon(${splitX}px 0, 100% 0, 100% 100%, ${splitX}px 100%)`;
}

function updateSplitPosition() {
  if (!processedData) return;
  const { width } = processedData;
  const splitX = width * splitPos;
  splitDivider.style.left = `${splitX}px`;
  clipSplitCanvas();
}

// Interactive Palette UI
function buildPaletteUI() {
  if (!processedData) return;
  const { palette, regions } = processedData;

  paletteCountBadge.textContent = `${palette.length} Farben`;
  paletteList.innerHTML = '';

  palette.forEach(col => {
    const item = document.createElement('div');
    item.className = 'palette-item';
    item.dataset.number = col.number;

    if (activeHighlightNumber === col.number) {
      item.classList.add('active');
    }

    item.innerHTML = `
      <div class="palette-num">${col.number}</div>
      <div class="palette-swatch" style="background-color: ${col.hex};"></div>
      <div class="palette-meta">
        <span class="palette-name">${col.name}</span>
        <span class="palette-code">Code ${col.code}</span>
      </div>
      <div class="palette-stats">
        <div>${col.regionCount} F.</div>
        <small style="color:var(--text-muted);">${col.areaPercent}%</small>
      </div>
    `;

    item.addEventListener('click', () => {
      if (activeHighlightNumber === col.number) {
        clearHighlight();
      } else {
        setHighlightNumber(col.number);
      }
    });

    paletteList.appendChild(item);
  });
}

function setHighlightNumber(num) {
  activeHighlightNumber = num;
  const colObj = processedData.palette.find(p => p.number === num);

  // Update palette active states
  document.querySelectorAll('.palette-item').forEach(item => {
    if (parseInt(item.dataset.number, 10) === num) {
      item.classList.add('active');
      item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
      item.classList.remove('active');
    }
  });

  // Show banner
  if (colObj) {
    highlightBanner.style.display = 'flex';
    highlightBannerText.innerHTML = `
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${colObj.hex};margin-right:6px;"></span>
      Farbe #${colObj.number} (${colObj.name}, Code ${colObj.code}) hervorgehoben
    `;
  }

  renderCurrentView();
}

function clearHighlight() {
  activeHighlightNumber = null;
  highlightBanner.style.display = 'none';
  document.querySelectorAll('.palette-item').forEach(item => item.classList.remove('active'));
  renderCurrentView();
}

// Tooltip on Hover
function handleCanvasHover(e) {
  if (!processedData) return;

  const rect = mainCanvas.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  const screenY = e.clientY - rect.top;

  const imgX = Math.floor(screenX / scale);
  const imgY = Math.floor(screenY / scale);

  if (imgX < 0 || imgX >= processedData.width || imgY < 0 || imgY >= processedData.height) {
    canvasTooltip.style.display = 'none';
    return;
  }

  const pIdx = imgY * processedData.width + imgX;
  const regId = processedData.labels[pIdx];
  const reg = processedData.regions[regId];

  if (!reg) {
    canvasTooltip.style.display = 'none';
    return;
  }

  const col = processedData.palette.find(p => p.number === reg.number);
  if (!col) return;

  canvasTooltip.style.display = 'flex';
  canvasTooltip.style.left = `${e.clientX - canvasStage.getBoundingClientRect().left}px`;
  canvasTooltip.style.top = `${e.clientY - canvasStage.getBoundingClientRect().top}px`;

  tooltipSwatch.style.backgroundColor = col.hex;
  tooltipTitle.textContent = `#${col.number} ${col.name}`;
  tooltipSub.textContent = `Amsterdam Code ${col.code} • Fläche ${reg.area} px`;
}

function handleCanvasClick(e) {
  if (!processedData || isPanning) return;

  const rect = mainCanvas.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  const screenY = e.clientY - rect.top;

  const imgX = Math.floor(screenX / scale);
  const imgY = Math.floor(screenY / scale);

  if (imgX >= 0 && imgX < processedData.width && imgY >= 0 && imgY < processedData.height) {
    const pIdx = imgY * processedData.width + imgX;
    const regId = processedData.labels[pIdx];
    const reg = processedData.regions[regId];
    if (reg) {
      setHighlightNumber(reg.number);
    }
  }
}

// Pan & Zoom
function handleWheel(e) {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
  const rect = canvasStage.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const newScale = Math.max(0.15, Math.min(8.0, scale * zoomFactor));

  // Zoom centered on mouse
  panX = mouseX - (mouseX - panX) * (newScale / scale);
  panY = mouseY - (mouseY - panY) * (newScale / scale);
  scale = newScale;

  applyTransform();
}

function handleMouseDown(e) {
  if (e.target.closest('.split-divider')) return;

  isPanning = true;
  startPanX = e.clientX - panX;
  startPanY = e.clientY - panY;
  canvasStage.classList.add('grabbing');
}

function handleMouseMove(e) {
  if (isDraggingSplit && processedData) {
    const rect = mainCanvas.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / (mainCanvas.width * scale);
    splitPos = Math.max(0.05, Math.min(0.95, relX));
    updateSplitPosition();
    return;
  }

  if (isPanning) {
    panX = e.clientX - startPanX;
    panY = e.clientY - startPanY;
    applyTransform();
  }
}

function handleMouseUp() {
  isPanning = false;
  isDraggingSplit = false;
  canvasStage.classList.remove('grabbing');
}

function setZoom(newScale) {
  scale = Math.max(0.15, Math.min(8.0, newScale));
  applyTransform();
}

function zoomToFit() {
  if (!mainCanvas.width) return;
  const stageW = canvasStage.clientWidth - 40;
  const stageH = canvasStage.clientHeight - 40;

  const scaleW = stageW / mainCanvas.width;
  const scaleH = stageH / mainCanvas.height;
  scale = Math.min(scaleW, scaleH, 1.0);

  panX = (canvasStage.clientWidth - mainCanvas.width * scale) / 2;
  panY = (canvasStage.clientHeight - mainCanvas.height * scale) / 2;

  applyTransform();
}

function applyTransform() {
  canvasWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  zoomLevelText.textContent = `${Math.round(scale * 100)}%`;
}

// Export Functions
function exportPng() {
  if (!processedData) return;
  // Render template to offscreen canvas without highlights
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = processedData.width;
  exportCanvas.height = processedData.height;
  const ctx = exportCanvas.getContext('2d');
  renderTemplate(ctx, null);

  const link = document.createElement('a');
  link.download = 'malen-nach-zahlen-vorlage.png';
  link.href = exportCanvas.toDataURL('image/png');
  link.click();
}

function exportPreviewPng() {
  if (!processedData) return;
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = processedData.width;
  exportCanvas.height = processedData.height;
  const ctx = exportCanvas.getContext('2d');
  renderPreview(ctx, null);

  const link = document.createElement('a');
  link.download = 'malen-nach-zahlen-vorschau.png';
  link.href = exportCanvas.toDataURL('image/png');
  link.click();
}

function exportSvg() {
  if (!processedData) return;
  const { width, height, boundaries, regions } = processedData;

  let svg = `<?xml version="1.0" standalone="no"?>\n`;
  svg += `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">\n`;
  svg += `  <rect width="100%" height="100%" fill="#ffffff" />\n`;

  // Draw template canvas as background raster image or paths
  const templateCanvas = document.createElement('canvas');
  templateCanvas.width = width;
  templateCanvas.height = height;
  renderTemplate(templateCanvas.getContext('2d'), null);

  const dataUrl = templateCanvas.toDataURL('image/png');
  svg += `  <image href="${dataUrl}" width="${width}" height="${height}" />\n`;

  // Overlay clean vector text for all numbers
  svg += `  <g id="numbers" font-family="-apple-system, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central" fill="#1e293b">\n`;
  for (let i = 0; i < regions.length; i++) {
    const reg = regions[i];
    if (reg.radius >= 3) {
      const fSize = Math.max(7, Math.min(18, Math.round(reg.radius * 1.35)));
      svg += `    <text x="${reg.x}" y="${reg.y}" font-size="${fSize}">${reg.number}</text>\n`;
    }
  }
  svg += `  </g>\n`;
  svg += `</svg>`;

  const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = 'malen-nach-zahlen-vektor.svg';
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
}

function exportPdf() {
  if (!processedData) return;

  const { jsPDF } = window.jspdf || {};
  if (!jsPDF) {
    alert('PDF-Bibliothek lädt noch oder ist nicht verfügbar.');
    return;
  }

  const format = selectPaperFormat.value;
  let orientation = 'l';
  let unit = 'mm';
  let pdfWidth = 297;
  let pdfHeight = 210;

  if (format === 'a3') {
    pdfWidth = 420;
    pdfHeight = 297;
  } else if (format === 'canvas6040') {
    pdfWidth = 600;
    pdfHeight = 400;
  }

  const doc = new jsPDF({
    orientation,
    unit,
    format: [pdfWidth, pdfHeight]
  });

  const { width, height, palette } = processedData;

  // --- Page 1: Malvorlage (Linien & Zahlen) ---
  const canvasTemplate = document.createElement('canvas');
  canvasTemplate.width = width;
  canvasTemplate.height = height;
  renderTemplate(canvasTemplate.getContext('2d'), null);

  const margin = 12;
  const availW = pdfWidth - margin * 2;
  const availH = pdfHeight - margin * 2 - 10;
  const ratio = Math.min(availW / width, availH / height);
  const drawW = width * ratio;
  const drawH = height * ratio;
  const drawX = margin + (availW - drawW) / 2;
  const drawY = margin + 8;

  // Title
  doc.setFontSize(14);
  doc.text('Malen nach Zahlen – Vorlage', margin, margin);
  doc.setFontSize(9);
  doc.setTextColor(100);
  doc.text(`Amsterdam Standard Series • ${palette.length} Farben`, pdfWidth - margin, margin, { align: 'right' });

  // Border & Canvas Image
  doc.setDrawColor(200);
  doc.rect(drawX, drawY, drawW, drawH);
  doc.addImage(canvasTemplate.toDataURL('image/png'), 'PNG', drawX, drawY, drawW, drawH);

  // --- Page 2: Farblegende ---
  doc.addPage([pdfWidth, pdfHeight], orientation);
  doc.setFontSize(16);
  doc.setTextColor(20);
  doc.text('Farblegende – Amsterdam Standard Series', margin, margin + 4);
  doc.setFontSize(9);
  doc.setTextColor(100);
  doc.text('Mit Farbfeldern zum Farbabstrich und Amsterdam-Farbnummern', margin, margin + 10);

  // Render grid of palette items
  const cols = pdfWidth > 350 ? 4 : 3;
  const colWidth = (pdfWidth - margin * 2) / cols;
  const rowHeight = 18;
  const startY = margin + 18;

  palette.forEach((col, idx) => {
    const c = idx % cols;
    const r = Math.floor(idx / cols);
    const x = margin + c * colWidth;
    const y = startY + r * rowHeight;

    if (y + rowHeight > pdfHeight - margin) return; // fits on page

    // Color Swatch Box
    doc.setFillColor(col.rgb[0], col.rgb[1], col.rgb[2]);
    doc.setDrawColor(120);
    doc.rect(x + 12, y, 14, 12, 'FD');

    // Number
    doc.setFontSize(12);
    doc.setTextColor(0);
    doc.text(String(col.number), x + 5, y + 8, { align: 'center' });

    // Name & Code
    doc.setFontSize(9);
    doc.setTextColor(20);
    doc.text(col.name, x + 30, y + 5);
    doc.setFontSize(8);
    doc.setTextColor(100);
    doc.text(`Code: ${col.code} • ${col.regionCount} Flächen (${col.areaPercent}%)`, x + 30, y + 10);

    // Dab area / checkbox
    doc.setDrawColor(180);
    doc.rect(x + colWidth - 14, y + 1, 10, 10);
  });

  // --- Page 3: Farbige fertige Vorschau ---
  doc.addPage([pdfWidth, pdfHeight], orientation);
  const canvasPreview = document.createElement('canvas');
  canvasPreview.width = width;
  canvasPreview.height = height;
  renderPreview(canvasPreview.getContext('2d'), null);

  doc.setFontSize(14);
  doc.setTextColor(20);
  doc.text('Farbige Vorschau (Ausgemaltes Bild)', margin, margin);

  doc.addImage(canvasPreview.toDataURL('image/png'), 'PNG', drawX, drawY, drawW, drawH);

  doc.save('malen-nach-zahlen-druckvorlage.pdf');
}

// Start app
window.addEventListener('DOMContentLoaded', init);
