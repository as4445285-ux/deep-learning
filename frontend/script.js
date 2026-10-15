const form = document.getElementById('captionForm');
const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const previewImage = document.getElementById('previewImage');
const imagePreview = document.getElementById('imagePreview');
const imageInfo = document.getElementById('imageInfo');
const clearImageBtn = document.getElementById('clearImageBtn');
const temperature = document.getElementById('temperature');
const maxLen = document.getElementById('maxLen');
const temperatureValue = document.getElementById('temperatureValue');
const maxLenValue = document.getElementById('maxLenValue');
const generateBtn = document.getElementById('generateBtn');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const captionResult = document.getElementById('captionResult');
const captionText = document.getElementById('captionText');
const elapsedMeta = document.getElementById('elapsedMeta');
const dimsMeta = document.getElementById('dimsMeta');
const timeMeta = document.getElementById('timeMeta');
const demoBanner = document.getElementById('demoBanner');
const errorBox = document.getElementById('errorBox');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const historyGrid = document.getElementById('historyGrid');

let selectedFile = null;

function showError(message) {
  if (!message) {
    errorBox.style.display = 'none';
    errorBox.textContent = '';
    return;
  }

  errorBox.textContent = message;
  errorBox.style.display = 'block';
}

function updateSliderLabels() {
  temperatureValue.textContent = Number(temperature.value).toFixed(1);
  maxLenValue.textContent = maxLen.value;
}

temperature.addEventListener('input', updateSliderLabels);
maxLen.addEventListener('input', updateSliderLabels);
updateSliderLabels();

function setPreviewFromFile(file) {
  if (!file) {
    imagePreview.style.display = 'none';
    selectedFile = null;
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    imagePreview.style.display = 'block';
    imageInfo.textContent = `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  };
  reader.readAsDataURL(file);
  selectedFile = file;
}

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  setPreviewFromFile(file);
});

clearImageBtn.addEventListener('click', () => {
  imageInput.value = '';
  setPreviewFromFile(null);
});

dropZone.addEventListener('click', () => imageInput.click());
dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('drag-over');
  const [file] = event.dataTransfer.files;
  if (file) {
    setPreviewFromFile(file);
  }
});

dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    imageInput.click();
  }
});

function setLoadingState(isLoading) {
  generateBtn.disabled = isLoading;
  generateBtn.innerHTML = isLoading
    ? '<span>Generating…</span>'
    : '<span>Generate caption</span>';
  loadingState.style.display = isLoading ? 'flex' : 'none';
  emptyState.style.display = isLoading || selectedFile ? 'none' : 'flex';
  if (!isLoading) {
    showError('');
  }
}

function renderHistory(entries) {
  if (!entries.length) {
    historyGrid.innerHTML = '<div class="history-empty">No previous captions yet.</div>';
    return;
  }

  historyGrid.innerHTML = entries
    .map((entry) => {
      const thumb = entry.thumbnail
        ? `data:image/jpeg;base64,${entry.thumbnail}`
        : 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="180"><rect width="100%" height="100%" fill="#0d1117"/><text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="24" font-family="Arial">Image</text></svg>');

      const stamp = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : 'recent';
      return `
        <div class="history-card" title="${entry.caption}">
          <img class="history-thumb" src="${thumb}" alt="${entry.filename || 'Captioned image'}" />
          <div class="history-body">
            <div class="history-caption">${entry.caption}</div>
            <div class="history-meta">
              <span>${entry.filename || 'Image'}</span>
              <span>${stamp}</span>
            </div>
          </div>
        </div>
      `;
    })
    .join('');
}

async function loadHistory() {
  try {
    const response = await fetch('/api/history');
    if (!response.ok) return;
    const items = await response.json();
    renderHistory(items);
  } catch (error) {
    console.error('Could not load recent captions.', error);
  }
}

async function clearHistory() {
  try {
    await fetch('/api/history', { method: 'DELETE' });
    await loadHistory();
  } catch (error) {
    console.error('Could not clear history.', error);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!selectedFile) {
    alert('Please select an image first.');
    return;
  }

  setLoadingState(true);
  captionResult.style.display = 'none';

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('temperature', Number(temperature.value).toFixed(1));
    formData.append('max_len', maxLen.value);

    const response = await fetch('/api/caption', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Caption generation failed.');
    }

    showError('');
    captionText.textContent = data.caption || 'No caption returned.';
    elapsedMeta.textContent = `${data.elapsed_ms} ms`;
    dimsMeta.textContent = `${data.width}×${data.height}`;
    timeMeta.textContent = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '--';
    demoBanner.style.display = data.demo_mode ? 'block' : 'none';

    captionResult.style.display = 'flex';
    emptyState.style.display = 'none';
    await loadHistory();
  } catch (error) {
    console.error(error);
    captionResult.style.display = 'flex';
    captionText.textContent = 'Caption unavailable.';
    showError(error.message || 'Unexpected error while generating caption.');
  } finally {
    setLoadingState(false);
  }
});

clearHistoryBtn.addEventListener('click', clearHistory);

loadHistory();
