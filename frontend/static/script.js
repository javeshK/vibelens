/**
 * SoundMatch AI — Frontend JavaScript
 * Handles image upload, cropping, API calls, and results display
 */

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════

const state = {
  imageFile: null,
  croppedData: null,
  selectedLang: "english",
  cropper: null,
};

// ═══════════════════════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elements = {
  // Upload
  dropZone: $("#dropZone"),
  dropContent: $("#dropContent"),
  dropPreview: $("#dropPreview"),
  fileInput: $("#fileInput"),
  
  // Crop
  cropToolbar: $("#cropToolbar"),
  openCropModal: $("#openCropModal"),
  cropStatus: $("#cropStatus"),
  clearCrop: $("#clearCrop"),
  cropModalOverlay: $("#cropModalOverlay"),
  cropModalClose: $("#cropModalClose"),
  cropImage: $("#cropImage"),
  cropResetBtn: $("#cropResetBtn"),
  cropConfirmBtn: $("#cropConfirmBtn"),
  
  // Language
  langTabs: $$(".lang-tab"),
  
  // Analyse
  analyseBtn: $("#analyseBtn"),
  btnContent: $("#btnContent"),
  btnIcon: $("#btnIcon"),
  btnText: $("#btnText"),
  btnLoader: $("#btnLoader"),
  
  // Loading
  aiLoading: $("#aiLoading"),
  loadingTitle: $("#loadingTitle"),
  loadingSub: $("#loadingSub"),
  aiSteps: $$(".ai-step"),
  
  // Results
  resultsSection: $("#resultsSection"),
  summaryMood: $("#summaryMood"),
  summarySecondary: $("#summarySecondary"),
  summaryContext: $("#summaryContext"),
  summaryMetrics: $("#summaryMetrics"),
  aestheticCard: $("#aestheticCard"),
  signalBars: $("#signalBars"),
  faceDetectionCard: $("#faceDetectionCard"),
  resultLangLabel: $("#resultLangLabel"),
  resultsCount: $("#resultsCount"),
  cardsList: $("#cardsList"),
  resetBtn: $("#resetBtn"),
  
  // Toast
  toast: $("#toast"),
  
  // Theme
  themeToggle: $("#themeToggle"),
  themeIcon: $("#themeIcon"),
};

// ═══════════════════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  initUpload();
  initCrop();
  initLanguage();
  initAnalyse();
  initReset();
  initTheme();
  initSmoothScroll();
});

// ═══════════════════════════════════════════════════════════════════════════
// Upload Handling
// ═══════════════════════════════════════════════════════════════════════════

function initUpload() {
  const { dropZone, fileInput, dropPreview, dropContent } = elements;
  
  // Click to browse
  dropZone.addEventListener("click", (e) => {
    if (e.target.tagName !== "INPUT") {
      fileInput.click();
    }
  });
  
  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });
  
  // Drag & drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });
  
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
  
  // Keyboard accessibility
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Please upload an image file", "error");
    return;
  }
  
  if (file.size > 15 * 1024 * 1024) {
    showToast("Image must be under 15 MB", "error");
    return;
  }
  
  state.imageFile = file;
  state.croppedData = null;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    elements.dropPreview.src = e.target.result;
    elements.dropPreview.classList.remove("hidden");
    elements.dropContent.classList.add("hidden");
    elements.cropToolbar.classList.remove("hidden");
    elements.clearCrop.classList.add("hidden");
    elements.cropStatus.textContent = "No crop selected";
    updateAnalyseBtn();
  };
  reader.readAsDataURL(file);
}

// ═══════════════════════════════════════════════════════════════════════════
// Cropping
// ═══════════════════════════════════════════════════════════════════════════

function initCrop() {
  const {
    openCropModal, cropModalOverlay, cropModalClose,
    cropResetBtn, cropConfirmBtn, clearCrop
  } = elements;
  
  // Open modal
  openCropModal.addEventListener("click", () => {
    if (!state.imageFile) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      elements.cropImage.src = e.target.result;
      elements.cropModalOverlay.classList.remove("hidden");
      
      // Initialize Cropper.js
      if (state.cropper) state.cropper.destroy();
      state.cropper = new Cropper(elements.cropImage, {
        aspectRatio: NaN,
        viewMode: 1,
        autoCropArea: 0.8,
        responsive: true,
      });
    };
    reader.readAsDataURL(state.imageFile);
  });
  
  // Close modal
  cropModalClose.addEventListener("click", closeCropModal);
  cropModalOverlay.addEventListener("click", (e) => {
    if (e.target === cropModalOverlay) closeCropModal();
  });
  
  // Reset crop
  cropResetBtn.addEventListener("click", () => {
    if (state.cropper) state.cropper.reset();
  });
  
  // Confirm crop
  cropConfirmBtn.addEventListener("click", () => {
    if (!state.cropper) return;
    
    const canvas = state.cropper.getCroppedCanvas({
      maxWidth: 1200,
      maxHeight: 1200,
    });
    
    state.croppedData = canvas.toDataURL("image/jpeg", 0.9);
    elements.cropStatus.textContent = "Crop applied";
    elements.clearCrop.classList.remove("hidden");
    closeCropModal();
  });
  
  // Clear crop
  clearCrop.addEventListener("click", () => {
    state.croppedData = null;
    elements.cropStatus.textContent = "No crop selected";
    elements.clearCrop.classList.add("hidden");
  });
}

function closeCropModal() {
  elements.cropModalOverlay.classList.add("hidden");
  if (state.cropper) {
    state.cropper.destroy();
    state.cropper = null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Language Selection
// ═══════════════════════════════════════════════════════════════════════════

function initLanguage() {
  elements.langTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      elements.langTabs.forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-pressed", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-pressed", "true");
      state.selectedLang = tab.dataset.lang;
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Analysis & Recommendation
// ═══════════════════════════════════════════════════════════════════════════

function initAnalyse() {
  elements.analyseBtn.addEventListener("click", runAnalysis);
}

function updateAnalyseBtn() {
  elements.analyseBtn.disabled = !state.imageFile;
}

async function runAnalysis() {
  if (!state.imageFile) return;
  
  // Show loading
  showLoading(true);
  updateLoadingStep(1);
  
  try {
    // Prepare image data
    const imageData = state.croppedData || await fileToBase64(state.imageFile);
    
    // Call API
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageData,
        language: state.selectedLang,
      }),
    });
    
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Analysis failed");
    }
    
    const result = await response.json();
    
    updateLoadingStep(2);
    await delay(300);
    updateLoadingStep(3);
    await delay(300);
    updateLoadingStep(4);
    await delay(300);
    
    // Display results
    showLoading(false);
    displayResults(result);
    
  } catch (err) {
    showLoading(false);
    showToast(err.message, "error");
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Loading Display
// ═══════════════════════════════════════════════════════════════════════════

function showLoading(show) {
  if (show) {
    elements.aiLoading.classList.remove("hidden");
    elements.resultsSection.classList.add("hidden");
  } else {
    elements.aiLoading.classList.add("hidden");
  }
}

function updateLoadingStep(step) {
  elements.aiSteps.forEach((el, i) => {
    el.classList.toggle("active", i + 1 <= step);
  });
  
  const titles = [
    "Analysing your image…",
    "Reading colour psychology & mood signals",
    "Matching with trending songs",
    "Scoring and ranking results",
  ];
  elements.loadingTitle.textContent = titles[step - 1] || titles[0];
}

// ═══════════════════════════════════════════════════════════════════════════
// Results Display
// ═══════════════════════════════════════════════════════════════════════════

function displayResults(data) {
  const { analysis, recommendations } = data;
  
  // Update language label
  const langLabels = { english: "English", hindi: "Hindi", punjabi: "Punjabi" };
  elements.resultLangLabel.textContent = langLabels[state.selectedLang] || "English";
  
  // Summary (backend sends "mood" as string, "context" as string)
  elements.summaryMood.textContent = analysis.mood || "—";
  elements.summaryContext.textContent = analysis.context || "";
  
  // Secondary moods (backend sends "secondary_moods" as array)
  if (analysis.secondary_moods?.length) {
    elements.summarySecondary.innerHTML = analysis.secondary_moods
      .map((m) => `<span class="mood-pill-sm">${m}</span>`)
      .join("");
  } else {
    elements.summarySecondary.innerHTML = "";
  }
  
  // Metrics (backend sends "color" not "color_metrics")
  const metrics = analysis.color || {};
  elements.summaryMetrics.innerHTML = `
    <div class="metric"><span class="metric-val">${fmt(metrics.brightness)}</span><span class="metric-lbl">Brightness</span></div>
    <div class="metric"><span class="metric-val">${fmt(metrics.saturation)}</span><span class="metric-lbl">Saturation</span></div>
    <div class="metric"><span class="metric-val">${fmt(metrics.contrast)}</span><span class="metric-lbl">Contrast</span></div>
    <div class="metric"><span class="metric-val">${fmt(metrics.sharpness)}</span><span class="metric-lbl">Sharpness</span></div>
  `;
  
  // Aesthetic card (backend sends "aesthetic" not "aesthetic_score")
  const aesthetic = analysis.aesthetic || {};
  elements.aestheticCard.innerHTML = `
    <div class="aesthetic-grade">${aesthetic.grade || "—"}</div>
    <div class="aesthetic-label">Aesthetic Score</div>
    <div class="aesthetic-bar"><div class="aesthetic-fill" style="width: ${(aesthetic.overall || 0) * 100}%"></div></div>
  `;
  
  // Signal bars
  const signals = [
    { label: "Warmth", value: metrics.warm_ratio || 0 },
    { label: "Cool tones", value: metrics.cool_ratio || 0 },
    { label: "Darkness", value: metrics.dark_ratio || 0 },
  ];
  elements.signalBars.innerHTML = signals
    .map((s) => `
      <div class="signal-row">
        <span class="signal-label">${s.label}</span>
        <div class="signal-track"><div class="signal-fill" style="width: ${s.value * 100}%"></div></div>
        <span class="signal-val">${Math.round(s.value * 100)}%</span>
      </div>
    `)
    .join("");
  
  // Song cards (backend uses "song_name" and "final_score")
  elements.resultsCount.textContent = `${recommendations.length} songs matched`;
  elements.cardsList.innerHTML = recommendations.map((song, i) => `
    <div class="song-card" role="listitem">
      <div class="song-rank">${i + 1}</div>
      <div class="song-info">
        <div class="song-title">${song.song_name || song.title || "Unknown"}</div>
        <div class="song-artist">${song.artist}</div>
      </div>
      <div class="song-score">
        <span class="score-val">${Math.round(song.final_score || song.score || 0)}</span>
        <span class="score-label">match</span>
      </div>
      ${song.youtube_id ? `
        <a href="https://youtube.com/watch?v=${song.youtube_id}" 
           target="_blank" rel="noopener" class="song-yt" aria-label="Play on YouTube">
          ▶
        </a>
      ` : ""}
    </div>
  `).join("");
  
  // Show results
  elements.resultsSection.classList.remove("hidden");
  elements.resultsSection.scrollIntoView({ behavior: "smooth" });
}

function fmt(v) {
  return v != null ? Math.round(v * 100) + "%" : "—";
}

// ═══════════════════════════════════════════════════════════════════════════
// Reset
// ═══════════════════════════════════════════════════════════════════════════

function initReset() {
  elements.resetBtn.addEventListener("click", () => {
    state.imageFile = null;
    state.croppedData = null;
    
    elements.dropPreview.src = "";
    elements.dropPreview.classList.add("hidden");
    elements.dropContent.classList.remove("hidden");
    elements.cropToolbar.classList.add("hidden");
    elements.resultsSection.classList.add("hidden");
    elements.fileInput.value = "";
    
    updateAnalyseBtn();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Theme Toggle
// ═══════════════════════════════════════════════════════════════════════════

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    elements.themeIcon.textContent = "🌙";
  }
  
  elements.themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const newTheme = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    elements.themeIcon.textContent = newTheme === "dark" ? "🌙" : "☀";
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Smooth Scroll
// ═══════════════════════════════════════════════════════════════════════════

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      e.preventDefault();
      const target = $(anchor.getAttribute("href"));
      if (target) {
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════════════════════════════════

function showToast(message, type = "info") {
  elements.toast.textContent = message;
  elements.toast.className = `toast toast-${type} show`;
  
  setTimeout(() => {
    elements.toast.classList.remove("show");
  }, 4000);
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}