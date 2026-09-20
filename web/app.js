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
const checkKuwahara = document.getElementById('checkKuwahara');

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
const sliderImpasto = document.getElementById('sliderImpasto');
const valImpasto = document.getElementById('valImpasto');

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

// Presets (Balanced for real-world acrylic painting)
const PRESETS = {
  street: {
    colors: 28, minRegion: 140, smooth: true, kuwahara: true,
    gamma: 1.00, contrast: 1.02, saturation: 1.15, lineWidth: 1.0
  },
  portrait: {
    colors: 22, minRegion: 200, smooth: true, kuwahara: true,
    gamma: 0.95, contrast: 1.05, saturation: 1.08, lineWidth: 1.0
  },
  landscape: {
    colors: 26, minRegion: 260, smooth: true, kuwahara: true,
    gamma: 1.00, contrast: 1.00, saturation: 1.05, lineWidth: 1.0
  },
  popart: {
    colors: 14, minRegion: 300, smooth: true, kuwahara: false,
    gamma: 0.85, contrast: 1.25, saturation: 1.30, lineWidth: 1.5
  },
  beginner: {
    colors: 14, minRegion: 450, smooth: true, kuwahara: true,
    gamma: 1.00, contrast: 1.00, saturation: 1.00, lineWidth: 1.2
  },
  detailed: {
    colors: 32, minRegion: 160, smooth: true, kuwahara: true,
    gamma: 1.05, contrast: 1.05, saturation: 1.15, lineWidth: 1.0
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
      const grp = document.getElementById('groupImpasto');
      if (grp) grp.style.display = checkAcrylicEffect.checked ? 'flex' : 'none';
      if (processedData) renderCurrentView();
    });
  }
  if (sliderImpasto) {
    sliderImpasto.addEventListener('input', () => {
      if (valImpasto) valImpasto.textContent = `${Math.round(sliderImpasto.value * 100)}%`;
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
    document.querySelectorAll('.dropdown').forEach(d => {
      if (d !== btnExportMenu.parentElement) d.classList.remove('open');
    });
    btnExportMenu.parentElement.classList.toggle('open');
  });

  // Project menu
  const btnProjectMenu = document.getElementById('btnProjectMenu');
  if (btnProjectMenu) {
    btnProjectMenu.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('.dropdown').forEach(d => {
        if (d !== btnProjectMenu.parentElement) d.classList.remove('open');
      });
      btnProjectMenu.parentElement.classList.toggle('open');
    });
  }

  const btnSaveProject = document.getElementById('btnSaveProject');
  if (btnSaveProject) btnSaveProject.addEventListener('click', saveProject);

  const btnLoadProject = document.getElementById('btnLoadProject');
  const projectFileInput = document.getElementById('projectFileInput');
  if (btnLoadProject && projectFileInput) {
    btnLoadProject.addEventListener('click', () => projectFileInput.click());
    projectFileInput.addEventListener('change', handleProjectFileSelect);
  }

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
  if (checkKuwahara && p.kuwahara !== undefined) {
    checkKuwahara.checked = p.kuwahara;
  }

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
    kuwahara: checkKuwahara ? checkKuwahara.checked : true,
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
  const { width, height, polylines, boundaries, regions, palette } = processedData;
  const lineWidth = parseFloat(sliderLineWidth.value) || 1.0;
  const numScale = parseInt(sliderNumberScale.value, 10) / 100.0;

  // 1. Fill background white
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // 2. If a color is highlighted, paint its regions in gentle accent
  if (highlightNum !== null) {
    const colObj = palette.find(p => p.number === highlightNum);
    const rgb = colObj ? colObj.rgb : [254, 240, 138];
    const labels = processedData.labels;
    const imgData = ctx.getImageData(0, 0, width, height);
    const d = imgData.data;

    for (let i = 0; i < labels.length; i++) {
      const reg = regions[labels[i]];
      if (reg && reg.number === highlightNum) {
        const p = i * 4;
        d[p] = Math.round(255 * 0.35 + rgb[0] * 0.65);
        d[p + 1] = Math.round(255 * 0.35 + rgb[1] * 0.65);
        d[p + 2] = Math.round(255 * 0.35 + rgb[2] * 0.65);
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // 3. Draw Clean Anti-Aliased Vector Boundaries (Smooth Fineliner Look)
  if (polylines && polylines.length > 0) {
    ctx.save();
    ctx.beginPath();
    let offset = 0;
    while (offset < polylines.length) {
      const len = polylines[offset++];
      if (len < 2) {
        offset += len * 2;
        continue;
      }
      const x0 = polylines[offset++];
      const y0 = polylines[offset++];
      ctx.moveTo(x0, y0);
      for (let j = 1; j < len; j++) {
        ctx.lineTo(polylines[offset++], polylines[offset++]);
      }
    }
    ctx.strokeStyle = '#334155'; // Slate charcoal fineliner
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.restore();
  } else if (boundaries) {
    // Fallback raster boundaries if polylines unavailable
    const imgData = ctx.getImageData(0, 0, width, height);
    const d = imgData.data;
    const boundaryColor = [51, 65, 85];
    for (let i = 0; i < boundaries.length; i++) {
      if (boundaries[i] === 1) {
        const p = i * 4;
        d[p] = boundaryColor[0];
        d[p + 1] = boundaryColor[1];
        d[p + 2] = boundaryColor[2];
        d[p + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // 4. Draw Numbers at Polylabel Centers with Crisp White Halo (Never cut by lines)
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < regions.length; i++) {
    const reg = regions[i];
    const s = String(reg.number);

    // Enforce strict clearance: single digits radius >= 4px, double digits >= 6px
    const minR = s.length > 1 ? 6.0 : 4.0;
    if (reg.radius < minR) continue;

    const isTarget = highlightNum !== null && reg.number === highlightNum;
    const baseFontSize = Math.max(8, Math.min(16, Math.round(reg.radius * 1.25)));
    const finalSize = Math.round(baseFontSize * numScale);

    ctx.font = `600 ${finalSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;

    if (isTarget) {
      // Highlighted circle pin
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(reg.x, reg.y, finalSize * 0.9, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.fillText(s, reg.x, reg.y + 0.5);
    } else {
      // White protective halo around number so intersecting contour lines don't obscure the digits
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = Math.max(3, finalSize * 0.38);
      ctx.lineJoin = 'round';
      ctx.miterLimit = 2;
      ctx.strokeText(s, reg.x, reg.y + 0.5);

      ctx.fillStyle = highlightNum !== null ? '#94a3b8' : '#1e293b';
      ctx.fillText(s, reg.x, reg.y + 0.5);
    }
  }
  ctx.restore();
}

function renderPreview(ctx, highlightNum = null) {
  const { width, height, labels, regions, palette, boundaries, preprocessedPixels } = processedData;

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

  const isAcrylic = !checkAcrylicEffect || checkAcrylicEffect.checked;
  const strength = sliderImpasto ? parseFloat(sliderImpasto.value) : 1.0;

  for (let i = 0; i < labels.length; i++) {
    const regId = labels[i];
    const cOffset = regId * 3;
    const p = i * 4;

    const rPal = colorMap[cOffset];
    const gPal = colorMap[cOffset + 1];
    const bPal = colorMap[cOffset + 2];

    let deltaY = 0;
    if (isAcrylic && preprocessedPixels && highlightNum === null) {
      const rPre = preprocessedPixels[p];
      const gPre = preprocessedPixels[p + 1];
      const bPre = preprocessedPixels[p + 2];
      const yPre = 0.299 * rPre + 0.587 * gPre + 0.114 * bPre;
      const yPal = 0.299 * rPal + 0.587 * gPal + 0.114 * bPal;
      // Painterly form & reflection shading (+-18 levels, preserving palette color identity)
      deltaY = Math.max(-18, Math.min(18, (yPre - yPal) * 0.25));
    }

    d[p] = Math.max(0, Math.min(255, rPal + deltaY));
    d[p + 1] = Math.max(0, Math.min(255, gPal + deltaY));
    d[p + 2] = Math.max(0, Math.min(255, bPal + deltaY));
    d[p + 3] = 255;
  }

  // Soften pixel staircases along color boundaries for organic brush edges
  softenColorBoundaries(d, labels, width, height);

  // Optional Acrylic & Canvas Texture (Impasto, Pinselduktus & Leinwandgewebe)
  if (isAcrylic && highlightNum === null) {
    applyOrganicAcrylicTexture(d, width, height, boundaries, labels, regions, preprocessedPixels, strength);
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

// Anti-alias pixel staircases symmetrically so color fields meet like wet acrylic paint
function softenColorBoundaries(data, labels, width, height) {
  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOffset + x;
      const curr = labels[idx];

      const rL = labels[idx - 1];
      const rR = labels[idx + 1];
      const rT = labels[idx - width];
      const rB = labels[idx + width];

      if (rL !== curr || rR !== curr || rT !== curr || rB !== curr) {
        const p = idx * 4;
        let sumR = data[p] * 2, sumG = data[p + 1] * 2, sumB = data[p + 2] * 2;
        let count = 2;

        if (rL !== curr) { const pN = (idx - 1) * 4; sumR += data[pN]; sumG += data[pN + 1]; sumB += data[pN + 2]; count++; }
        if (rR !== curr) { const pN = (idx + 1) * 4; sumR += data[pN]; sumG += data[pN + 1]; sumB += data[pN + 2]; count++; }
        if (rT !== curr) { const pN = (idx - width) * 4; sumR += data[pN]; sumG += data[pN + 1]; sumB += data[pN + 2]; count++; }
        if (rB !== curr) { const pN = (idx + width) * 4; sumR += data[pN]; sumG += data[pN + 1]; sumB += data[pN + 2]; count++; }

        data[p] = Math.round(sumR / count);
        data[p + 1] = Math.round(sumG / count);
        data[p + 2] = Math.round(sumB / count);
      }
    }
  }
}

// Fast Separable Box Blur for Structure Tensor
function boxBlurFloat(input, w, h, radius) {
  const output = new Float32Array(w * h);
  const temp = new Float32Array(w * h);
  const winSize = 2 * radius + 1;

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let k = -radius; k <= radius; k++) {
      const px = Math.max(0, Math.min(w - 1, k));
      sum += input[row + px];
    }
    temp[row] = sum / winSize;

    for (let x = 1; x < w; x++) {
      const addX = Math.min(w - 1, x + radius);
      const subX = Math.max(0, x - radius - 1);
      sum += input[row + addX] - input[row + subX];
      temp[row + x] = sum / winSize;
    }
  }

  // Vertical pass
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let k = -radius; k <= radius; k++) {
      const py = Math.max(0, Math.min(h - 1, k));
      sum += temp[py * w + x];
    }
    output[x] = sum / winSize;

    for (let y = 1; y < h; y++) {
      const addY = Math.min(h - 1, y + radius);
      const subY = Math.max(0, y - radius - 1);
      sum += temp[addY * w + x] - temp[subY * w + x];
      output[y * w + x] = sum / winSize;
    }
  }

  return output;
}

// Authentic Acrylic Impasto & Flow-Guided Paint Shader (Reverse-Engineered from Reference Painting)
// Uses Structure Tensor image flow so strokes naturally follow 3D object contours
// (horizontal road/curb, vertical buildings/legs, perspective curves on cars),
// shuffled discrete physical brush strokes per region, and Amsterdam satin sheen
function applyOrganicAcrylicTexture(data, width, height, boundaries, labels, regions, preprocessedPixels, strength = 1.0) {
  const total = width * height;

  // 1. Compute guide luminance from preprocessed Kuwahara image (or current canvas data)
  const lum = new Float32Array(total);
  if (preprocessedPixels && preprocessedPixels.length === total * 4) {
    for (let i = 0; i < total; i++) {
      const p = i * 4;
      lum[i] = 0.299 * preprocessedPixels[p] + 0.587 * preprocessedPixels[p + 1] + 0.114 * preprocessedPixels[p + 2];
    }
  } else {
    for (let i = 0; i < total; i++) {
      const p = i * 4;
      lum[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
  }

  // 2. Gradients and Structure Tensor (Jxx, Jyy, Jxy)
  const jxx = new Float32Array(total);
  const jyy = new Float32Array(total);
  const jxy = new Float32Array(total);

  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOffset + x;
      const gx = (lum[idx + 1] - lum[idx - 1]) * 0.5;
      const gy = (lum[idx + width] - lum[idx - width]) * 0.5;
      jxx[idx] = gx * gx;
      jyy[idx] = gy * gy;
      jxy[idx] = gx * gy;
    }
  }

  // Smooth structure tensor components
  const sJxx = boxBlurFloat(jxx, width, height, 3);
  const sJyy = boxBlurFloat(jyy, width, height, 3);
  const sJxy = boxBlurFloat(jxy, width, height, 3);

  // 3. Fast Two-Pass Distance Transform to Region Boundaries (Chamfer 3-4 metric)
  // Used to model paint boundary ridges (Farbstau an den Rändern) and field centers
  const dist = new Float32Array(total);
  const INF = 9999.0;
  for (let i = 0; i < total; i++) {
    dist[i] = (boundaries && boundaries[i] === 1) ? 0.0 : INF;
  }

  // Forward pass (top-left to bottom-right)
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const idx = row + x;
      let d = dist[idx];
      if (d === 0.0) continue;
      if (x > 0) d = Math.min(d, dist[idx - 1] + 1.0);
      if (y > 0) {
        d = Math.min(d, dist[idx - width] + 1.0);
        if (x > 0) d = Math.min(d, dist[idx - width - 1] + 1.414);
        if (x < width - 1) d = Math.min(d, dist[idx - width + 1] + 1.414);
      }
      dist[idx] = d;
    }
  }

  // Backward pass (bottom-right to top-left)
  for (let y = height - 1; y >= 0; y--) {
    const row = y * width;
    for (let x = width - 1; x >= 0; x--) {
      const idx = row + x;
      let d = dist[idx];
      if (x < width - 1) d = Math.min(d, dist[idx + 1] + 1.0);
      if (y < height - 1) {
        d = Math.min(d, dist[idx + width] + 1.0);
        if (x < width - 1) d = Math.min(d, dist[idx + width + 1] + 1.414);
        if (x > 0) d = Math.min(d, dist[idx + width - 1] + 1.414);
      }
      dist[idx] = d;
    }
  }

  // 4. Initialize Height Map with Canvas Linen Weave Base and Boundary Paint Ridges
  const H = new Float32Array(total);
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      const idx = rowOffset + x;
      const dVal = dist[idx];

      // Canvas linen weave (subtle woven threads, pitch ~ 3.6px)
      const canvas = Math.sin(x * 1.745) * Math.sin(y * 1.745) * 0.32;

      // Authentic Acrylic Paint Boundary Ridge: paint builds up 1.5 - 3.5px from edge
      const edgeDiff = dVal - 2.2;
      const ridge = Math.exp(-(edgeDiff * edgeDiff) / 3.2) * 2.2 * strength;

      // Gentle paint dome toward region interior
      const fillDome = Math.min(1.0, dVal / 9.0) * 0.8 * strength;

      H[idx] = canvas + ridge + fillDome;
    }
  }

  // 5. Broad Organic Physical Brush Marks (Only where paint is actively brushed)
  const strokeSpacing = 22;
  const brushLen = 46;
  const brushRadius = 14;

  let seedRng = 42;
  function fastRand() {
    seedRng = (seedRng * 1664525 + 1013904223) | 0;
    return ((seedRng >>> 0) / 4294967296.0);
  }

  const strokes = [];
  for (let y = 0; y < height; y += strokeSpacing) {
    for (let x = 0; x < width; x += strokeSpacing) {
      const jx = Math.round(x + (fastRand() - 0.5) * strokeSpacing * 0.8);
      const jy = Math.round(y + (fastRand() - 0.5) * strokeSpacing * 0.8);
      if (jx >= 0 && jx < width && jy >= 0 && jy < height) {
        const idx = jy * width + jx;
        const r = labels ? labels[idx] : 0;
        strokes.push({ x: jx, y: jy, r: r });
      }
    }
  }

  // Shuffle strokes
  for (let i = strokes.length - 1; i > 0; i--) {
    const j = Math.floor(fastRand() * (i + 1));
    const temp = strokes[i];
    strokes[i] = strokes[j];
    strokes[j] = temp;
  }

  const handAngles = [0.45, 1.15, 2.10, 2.70];
  for (let i = 0; i < strokes.length; i++) {
    const st = strokes[i];
    const sx = st.x;
    const sy = st.y;
    const r = st.r;
    const idx = sy * width + sx;

    const reg = (regions && r < regions.length) ? regions[r] : null;
    if (reg && reg.area && reg.area < 35) continue; // Tiny fields remain clean and smooth

    const regElong = reg ? (reg.elong || 1.0) : 1.0;
    const regAngle = reg ? reg.angle : 0;

    const xx = sJxx[idx], yy = sJyy[idx], xy = sJxy[idx];
    const coherence = Math.sqrt((xx - yy) * (xx - yy) + 4 * xy * xy) / (xx + yy + 1e-4);

    let baseAngle;
    if (reg && regElong > 1.25) {
      baseAngle = regAngle;
    } else if (coherence > 0.22) {
      baseAngle = 0.5 * Math.atan2(2 * xy, xx - yy) + 1.5708;
    } else {
      baseAngle = reg ? reg.angle : handAngles[(r * 7) % handAngles.length];
    }

    const theta = baseAngle + (fastRand() - 0.5) * 0.12;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    const fitRadius = reg ? Math.max(5.0, Math.min(brushRadius, (reg.radius || 10.0) * 1.2)) : brushRadius;
    const hl = brushLen * 0.5 * (0.9 + fastRand() * 0.3);
    const hw = fitRadius * (0.9 + fastRand() * 0.25);

    const maxDim = Math.ceil(Math.sqrt(hl * hl + hw * hw));
    const x0 = Math.max(0, sx - maxDim);
    const x1 = Math.min(width - 1, sx + maxDim);
    const y0 = Math.max(0, sy - maxDim);
    const y1 = Math.min(height - 1, sy + maxDim);

    const pSeed = idx * 4;
    const lumBase = (0.299 * data[pSeed] + 0.587 * data[pSeed + 1] + 0.114 * data[pSeed + 2]) / 255.0;
    const nuance = (fastRand() * 10.0 - 5.0) * strength;

    for (let py = y0; py <= y1; py++) {
      const rowOffset = py * width;
      for (let px = x0; px <= x1; px++) {
        const pIdx = rowOffset + px;
        if (labels && labels[pIdx] !== r) continue;

        const dx = px - sx;
        const dy = py - sy;
        const u = dx * cosT + dy * sinT;
        const v = -dx * sinT + dy * cosT;

        const normU = Math.abs(u) / hl;
        const normV = Math.abs(v) / hw;

        if (normU <= 1.0 && normV <= 1.0) {
          const wu = Math.cos(normU * 1.5708);
          const wv = Math.cos(normV * 1.5708);
          const strokeMask = wu * wv;

          // Broad, smooth bristle drag (spacing ~ 5px)
          const bristle = 0.50 * Math.sin(v * 1.25) + 0.30 * Math.sin(v * 0.70 + 1.1);

          // Physical stroke impasto
          const strokeH = strokeMask * (1.2 + bristle * 0.6) * strength;
          H[pIdx] = Math.max(H[pIdx], H[pIdx] * 0.4 + strokeH);

          // Subtle pigment nuance inside stroke (smooth color body)
          const alpha = strokeMask * 0.30;
          const deltaCol = (nuance + bristle * 3.0) * strength;
          const p = pIdx * 4;
          data[p] = Math.max(0, Math.min(255, data[p] * (1.0 - alpha) + (data[p] + deltaCol) * alpha));
          data[p + 1] = Math.max(0, Math.min(255, data[p + 1] * (1.0 - alpha) + (data[p + 1] + deltaCol) * alpha));
          data[p + 2] = Math.max(0, Math.min(255, data[p + 2] * (1.0 - alpha) + (data[p + 2] + deltaCol) * alpha));
        }
      }
    }
  }

  // 6. Directional Satin Lighting with Deep Black Protection
  const scaleH = 0.42 * strength;
  const lx = -0.55, ly = -0.65, lz = 0.52;
  const invL = 1.0 / Math.sqrt(lx * lx + ly * ly + lz * lz);
  const nLx = lx * invL, nLy = ly * invL, nLz = lz * invL;

  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOffset + x;
      const p = idx * 4;

      const dhdx = (H[idx + 1] - H[idx - 1]) * (scaleH * 0.5);
      const dhdy = (H[idx + width] - H[idx - width]) * (scaleH * 0.5);

      const invN = 1.0 / Math.sqrt(dhdx * dhdx + dhdy * dhdy + 1.0);
      const nx = -dhdx * invN;
      const ny = -dhdy * invN;
      const nz = 1.0 * invN;

      const nDotL = nx * nLx + ny * nLy + nz * nLz;

      // Dark protection: deep blacks (lamp black) retain pure depth without grey washing
      const lumVal = (0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]) / 255.0;
      const lumFactor = Math.max(0.0, Math.min(1.0, (lumVal - 0.03) / 0.20));

      const diffuse = (nDotL - 0.48) * 34.0 * strength * lumFactor;

      // Crisp specular reflections along paint ridge crests and car curves
      let spec = 0;
      if (nDotL > 0) {
        const rz = Math.max(0, 2 * nDotL * nz - nLz);
        spec = Math.pow(rz, 10) * 46.0 * strength;
      }

      data[p] = Math.max(0, Math.min(255, data[p] + diffuse + spec));
      data[p + 1] = Math.max(0, Math.min(255, data[p + 1] + diffuse + spec));
      data[p + 2] = Math.max(0, Math.min(255, data[p + 2] + diffuse + spec));
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
  const { width, height, polylines, regions } = processedData;
  const lineWidth = parseFloat(sliderLineWidth.value) || 1.0;

  let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  svg += `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">\n`;
  svg += `  <rect width="100%" height="100%" fill="#ffffff" />\n`;

  // 1. True SVG Vector Paths for Boundaries
  if (polylines && polylines.length > 0) {
    let pathsD = '';
    let offset = 0;
    while (offset < polylines.length) {
      const len = polylines[offset++];
      if (len < 2) {
        offset += len * 2;
        continue;
      }
      const x0 = polylines[offset++];
      const y0 = polylines[offset++];
      pathsD += `M${x0.toFixed(1)} ${y0.toFixed(1)}`;
      for (let j = 1; j < len; j++) {
        pathsD += ` L${polylines[offset++].toFixed(1)} ${polylines[offset++].toFixed(1)}`;
      }
      pathsD += ' ';
    }
    svg += `  <path d="${pathsD}" stroke="#334155" stroke-width="${lineWidth}" fill="none" stroke-linecap="round" stroke-linejoin="round" />\n`;
  }

  // 2. High-Legibility Numbers with Protective Stroke Halo
  svg += `  <g id="numbers" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central">\n`;
  for (let i = 0; i < regions.length; i++) {
    const reg = regions[i];
    const isSingleDigit = reg.number < 10;
    const minRadius = isSingleDigit ? 4.0 : 6.0;

    if (reg.radius >= minRadius) {
      const fSize = Math.max(7, Math.min(18, Math.round(reg.radius * 1.35)));
      svg += `    <text x="${reg.x}" y="${reg.y}" font-size="${fSize}" paint-order="stroke fill" stroke="#ffffff" stroke-width="2.5" stroke-linejoin="round" fill="#1e293b">${reg.number}</text>\n`;
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

// --- Project Save & Load (.mnz) ---
// Serializes the full calculation (labels, boundaries, polylines, palette, regions, settings, original image)
// into a lightweight, self-contained JSON/ArrayBuffer file so it can be restored instantly without recalculation.
function saveProject() {
  if (!processedData) {
    alert('Es ist noch keine Berechnung vorhanden, die gespeichert werden kann.');
    return;
  }

  // 1. Convert original image to Base64 data URL
  let originalImageBase64 = null;
  if (originalImage) {
    const c = document.createElement('canvas');
    c.width = originalImage.naturalWidth || originalImage.width;
    c.height = originalImage.naturalHeight || originalImage.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(originalImage, 0, 0);
    originalImageBase64 = c.toDataURL('image/jpeg', 0.95);
  }

  // 2. Helper to encode typed arrays as compact Base64
  function bufferToBase64(typedArray) {
    if (!typedArray) return null;
    const bytes = new Uint8Array(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
    let binary = '';
    const len = bytes.byteLength;
    const chunkSize = 0x8000;
    for (let i = 0; i < len; i += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + chunkSize, len)));
    }
    return btoa(binary);
  }

  const project = {
    version: '1.0',
    timestamp: new Date().toISOString(),
    width: processedData.width,
    height: processedData.height,
    settings: {
      colors: sliderColors.value,
      minRegionSize: sliderMinRegion.value,
      smooth: checkSmoothing.checked,
      kuwahara: checkKuwahara ? checkKuwahara.checked : true,
      gamma: sliderGamma.value,
      contrast: sliderContrast.value,
      saturation: sliderSaturation.value,
      lineWidth: sliderLineWidth.value,
      numberScale: sliderNumberScale.value,
      acrylicEffect: checkAcrylicEffect ? checkAcrylicEffect.checked : true,
      impasto: sliderImpasto ? sliderImpasto.value : '1.0',
      paperFormat: selectPaperFormat.value
    },
    palette: processedData.palette,
    regions: processedData.regions,
    // Binary buffers packed in Base64
    labels: bufferToBase64(processedData.labels),
    boundaries: bufferToBase64(processedData.boundaries),
    polylines: bufferToBase64(processedData.polylines),
    preprocessedPixels: bufferToBase64(processedData.preprocessedPixels),
    originalImage: originalImageBase64
  };

  const jsonStr = JSON.stringify(project);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const dateStr = new Date().toISOString().slice(0, 10);
  a.download = `malen-nach-zahlen-projekt-${dateStr}.mnz`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function handleProjectFileSelect(e) {
  if (!e.target.files || !e.target.files[0]) return;
  const file = e.target.files[0];
  const reader = new FileReader();

  reader.onload = function (evt) {
    try {
      const project = JSON.parse(evt.target.result);
      if (!project || !project.labels || !project.palette) {
        throw new Error('Ungültiges Projekt-Dateiformat.');
      }

      function base64ToUint8(b64) {
        if (!b64) return null;
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return bytes;
      }

      // Reconstruct Typed Arrays
      const labelsBytes = base64ToUint8(project.labels);
      const labels = new Uint32Array(labelsBytes.buffer);

      const boundariesBytes = base64ToUint8(project.boundaries);
      const boundaries = boundariesBytes ? new Uint8Array(boundariesBytes.buffer) : null;

      const polylinesBytes = base64ToUint8(project.polylines);
      const polylines = polylinesBytes ? new Float32Array(polylinesBytes.buffer) : null;

      const preprocBytes = base64ToUint8(project.preprocessedPixels);
      const preprocessedPixels = preprocBytes ? new Uint8ClampedArray(preprocBytes.buffer) : null;

      // Restore processedData
      processedData = {
        width: project.width,
        height: project.height,
        labels,
        boundaries,
        polylines,
        palette: project.palette,
        regions: project.regions,
        preprocessedPixels
      };

      // Restore UI Settings
      const s = project.settings || {};
      if (s.colors) { sliderColors.value = s.colors; valColors.textContent = s.colors; }
      if (s.minRegionSize) { sliderMinRegion.value = s.minRegionSize; valMinRegion.textContent = `${s.minRegionSize} px`; }
      if (s.smooth !== undefined) checkSmoothing.checked = s.smooth;
      if (s.kuwahara !== undefined && checkKuwahara) checkKuwahara.checked = s.kuwahara;
      if (s.gamma) { sliderGamma.value = s.gamma; valGamma.textContent = parseFloat(s.gamma).toFixed(2); }
      if (s.contrast) { sliderContrast.value = s.contrast; valContrast.textContent = parseFloat(s.contrast).toFixed(2); }
      if (s.saturation) { sliderSaturation.value = s.saturation; valSaturation.textContent = parseFloat(s.saturation).toFixed(2); }
      if (s.lineWidth) { sliderLineWidth.value = s.lineWidth; valLineWidth.textContent = `${s.lineWidth} px`; }
      if (s.numberScale) { sliderNumberScale.value = s.numberScale; valNumberScale.textContent = `${s.numberScale}%`; }
      if (s.acrylicEffect !== undefined && checkAcrylicEffect) {
        checkAcrylicEffect.checked = s.acrylicEffect;
        const grp = document.getElementById('groupImpasto');
        if (grp) grp.style.display = s.acrylicEffect ? 'flex' : 'none';
      }
      if (s.impasto && sliderImpasto) {
        sliderImpasto.value = s.impasto;
        if (valImpasto) valImpasto.textContent = `${Math.round(s.impasto * 100)}%`;
      }
      if (s.paperFormat && selectPaperFormat) selectPaperFormat.value = s.paperFormat;

      // Restore Original Image
      if (project.originalImage) {
        const img = new Image();
        img.onload = function () {
          originalImage = img;
          uncroppedImage = img;
          isCropped = false;
          emptyState.style.display = 'none';
          btnProcess.disabled = false;
          zoomToFit();
          renderCurrentView();
          buildPaletteUI();
        };
        img.src = project.originalImage;
      } else {
        emptyState.style.display = 'none';
        btnProcess.disabled = false;
        zoomToFit();
        renderCurrentView();
        buildPaletteUI();
      }

      // Reset file input so same file can be loaded again if desired
      e.target.value = '';
    } catch (err) {
      alert('Fehler beim Laden der Projektdatei: ' + err.message);
      console.error(err);
    }
  };

  reader.readAsText(file);
}

// Start app
window.addEventListener('DOMContentLoaded', init);
