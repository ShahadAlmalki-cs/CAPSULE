/**
 * CAPSULE — AI Medication Identification System
 * Main JavaScript · Taif University · Vision 2030
 */

'use strict';

// ── Sidebar toggle (mobile) ──────────────────────────────────────────────────
const sidebar   = document.querySelector('.sidebar');
const menuBtn   = document.querySelector('.menu-toggle');
const overlay   = document.createElement('div');
overlay.className = 'sidebar-overlay';
overlay.style.cssText = `
  display:none; position:fixed; inset:0;
  background:rgba(0,0,0,.4); z-index:150; cursor:pointer;
`;
document.body.appendChild(overlay);

if (menuBtn && sidebar) {
  menuBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    overlay.style.display = sidebar.classList.contains('open') ? 'block' : 'none';
  });
  overlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    overlay.style.display = 'none';
  });
}

// ── Auto-dismiss alerts ──────────────────────────────────────────────────────
document.querySelectorAll('.alert[data-auto-dismiss]').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity .4s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 400);
  }, 4000);
});

// ── Confirm delete buttons ───────────────────────────────────────────────────
document.querySelectorAll('[data-confirm]').forEach(btn => {
  btn.addEventListener('click', e => {
    if (!confirm(btn.dataset.confirm || 'Are you sure?')) e.preventDefault();
  });
});

// ============================================================
// IDENTIFY PAGE
// ============================================================
(function initIdentifyPage() {
  const form        = document.getElementById('identifyForm');
  if (!form) return;

  const textarea    = document.getElementById('medicationDesc');
  const resultsBox  = document.getElementById('resultsBox');
  const clearBtn    = document.getElementById('clearBtn');
  const charCount   = document.getElementById('charCount');
  const loadingEl   = document.getElementById('loadingIndicator');

  // Active filter state
  const filters = { color: null, shapes: [], packaging: [], intendedUse: [] };

  // ── Chip toggle logic ────────────────────────────────────────────────────
  function applyChipStyle(chip, isActive) {
    // Use both inline styles AND data-attribute for CSS selector backup
    chip.setAttribute('data-active', isActive ? 'true' : 'false');
    if (isActive) {
      chip.style.cssText = 'background:#2563eb !important;border-color:#2563eb !important;color:#fff !important;font-weight:700;box-shadow:0 3px 10px rgba(37,99,235,.5);transform:scale(1.08);';
    } else {
      chip.style.cssText = '';
    }
  }

  document.querySelectorAll('.chip[data-filter-group]').forEach(chip => {
    chip.addEventListener('click', () => {
      const group  = chip.dataset.filterGroup;
      const value  = chip.dataset.value;
      const single = chip.dataset.single === 'true';

      if (single) {
        // Single-select: check state BEFORE clearing siblings
        const wasActive = chip.classList.contains('active');
        const siblings = document.querySelectorAll(`.chip[data-filter-group="${group}"]`);
        siblings.forEach(s => { s.classList.remove('active'); applyChipStyle(s, false); });
        if (!wasActive) {
          chip.classList.add('active');
          applyChipStyle(chip, true);
        }
        filters[group] = wasActive ? null : value;
      } else {
        chip.classList.toggle('active');
        applyChipStyle(chip, chip.classList.contains('active'));
        if (!filters[group]) filters[group] = [];
        const idx = filters[group].indexOf(value);
        if (idx === -1) filters[group].push(value);
        else filters[group].splice(idx, 1);
      }
    });
  });

  // ── Color dropdown ───────────────────────────────────────────────────────
  const colorSelect = document.getElementById('colorSelect');
  if (colorSelect) {
    colorSelect.addEventListener('change', () => {
      filters.color = colorSelect.value || null;
    });
  }

  // ── Character counter + Arabic detection ────────────────────────────────
  const langIndicator = document.getElementById('langIndicator');
  if (textarea && charCount) {
    textarea.addEventListener('input', () => {
      const val = textarea.value;
      charCount.textContent = `${val.length} characters`;
      // Detect Arabic
      if (langIndicator) {
        const arabicChars = (val.match(/[\u0600-\u06ff]/g) || []).length;
        const isArabic = arabicChars > val.length * 0.2;
        langIndicator.textContent = isArabic ? 'عربي' : 'EN';
        langIndicator.style.color = isArabic ? '#1d4ed8' : '';
        langIndicator.style.borderColor = isArabic ? '#93c5fd' : '';
        langIndicator.style.background = isArabic ? '#dbeafe' : '';
      }
    });
  }

  // ── Clear ────────────────────────────────────────────────────────────────
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (textarea) textarea.value = '';
      if (charCount) charCount.textContent = '0 characters';
      if (colorSelect) {
        colorSelect.value = '';
        colorSelect.style.borderColor = '';
        colorSelect.style.background = '';
        colorSelect.style.fontWeight = '';
      }
      document.querySelectorAll('.chip.active').forEach(c => { c.classList.remove('active'); applyChipStyle(c, false); });
      Object.keys(filters).forEach(k => { filters[k] = Array.isArray(filters[k]) ? [] : null; });
      if (resultsBox) resultsBox.innerHTML = '';
    });
  }

  // ── Form submit → AI identify ────────────────────────────────────────────
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const desc = (textarea ? textarea.value : '').trim();

    // At least one search criterion must be provided
    const hasFilters = filters.color ||
                       (filters.shapes && filters.shapes.length) ||
                       (filters.packaging && filters.packaging.length) ||
                       (filters.intendedUse && filters.intendedUse.length);
    if (!desc && !hasFilters) {
      showAlert(resultsBox, 'Please enter a description or select at least one filter (color, shape, packaging, or intended use).', 'warning');
      return;
    }

    // Send description and filters SEPARATELY so the backend can
    // properly compute blended scores based on filter matching.
    setLoading(true, loadingEl);
    if (resultsBox) resultsBox.innerHTML = '';

    try {
      const resp = await fetch('/api/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: desc,
          top_k: 5,
          color: filters.color || '',
          shape: filters.shapes && filters.shapes.length ? filters.shapes[0] : '',
          packaging: filters.packaging && filters.packaging.length ? filters.packaging[0] : '',
          intended_use: filters.intendedUse && filters.intendedUse.length ? filters.intendedUse.join(' ') : '',
        }),
      });

      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const data = await resp.json();
      renderResults(data, resultsBox);
    } catch (err) {
      showAlert(resultsBox, 'An error occurred. Please try again.', 'danger');
      console.error(err);
    } finally {
      setLoading(false, loadingEl);
    }
  });

  // ── Render results ───────────────────────────────────────────────────────
  function renderResults(data, container) {
    if (!container) return;
    container.innerHTML = '';

    if (data.status === 'incomplete') {
      container.innerHTML = renderIncomplete(data.completeness);
      return;
    }

    if (data.status === 'no_match' || !data.results || data.results.length === 0) {
      container.innerHTML = renderNoMatch();
      return;
    }

    const [top, ...rest] = data.results;

    // Top match
    const topEl = document.createElement('div');
    topEl.innerHTML = renderTopMatch(top);
    container.appendChild(topEl);

    // Other suggestions
    if (rest.length > 0) {
      const suggestEl = document.createElement('div');
      suggestEl.className = 'mt-3';
      suggestEl.innerHTML = `
        <h5 class="fw-bold mb-2" style="font-size:.9rem;color:var(--text-secondary)">
          Other Suggestions
        </h5>
        <div class="suggestion-grid">
          ${rest.map(r => renderSuggestion(r)).join('')}
        </div>
      `;
      container.appendChild(suggestEl);
    }

    // Wire up confirm / view detail buttons
    container.querySelectorAll('[data-confirm-btn]').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.textContent = '✓ Confirmed';
        btn.disabled = true;
        btn.style.background = 'var(--success)';
        showToast(`${btn.dataset.name} confirmed.`, 'success');
      });
    });
  }

  function renderTopMatch(r) {
    const conf  = r.confidence_pct || Math.round(r.confidence_score * 100);
    const level = r.confidence_level || (conf >= 80 ? 'high' : conf >= 60 ? 'medium' : 'low');
    const form  = r.dosage_form || '';
    const color = r.color || '';
    const shape = r.shape || '';
    const cat   = r.category || '';
    const imgHtml = r.image_url
      ? `<div style="flex-shrink:0;width:130px;height:130px;border-radius:var(--radius-sm);
                     overflow:hidden;border:1.5px solid var(--border);background:var(--bg);
                     display:flex;align-items:center;justify-content:center">
           <img src="${escHtml(r.image_url)}" alt="${escHtml(r.medication)}"
                style="max-width:100%;max-height:100%;object-fit:contain;padding:6px"
                onerror="this.parentElement.style.display='none'">
         </div>`
      : '';
    return `
      <div class="result-card top-match mb-3">
        <div style="display:flex;gap:16px;align-items:flex-start">
          ${imgHtml}
          <div style="flex:1;min-width:0">
            <div class="d-flex align-center justify-between mb-2 flex-wrap gap-1">
              <div>
                <h4 style="margin:0;font-size:1.1rem;font-weight:800">${escHtml(r.medication)}</h4>
                <span style="font-size:.8rem;color:var(--text-secondary)">${escHtml(r.generic_name || '')}</span>
                ${r.strength ? `<span class="detail-tag ml-1">${escHtml(r.strength)}</span>` : ''}
              </div>
              <span class="confidence-badge ${level}">${conf}% confidence</span>
            </div>
            <div class="confidence-bar-wrap mb-2">
              <div class="confidence-bar ${level}" style="width:${conf}%"></div>
            </div>
            <div class="d-flex gap-1 flex-wrap mb-3" style="font-size:.8rem;color:var(--text-secondary)">
              ${form  ? `<span>Form: <strong>${escHtml(form)}</strong></span>` : ''}
              ${color ? `<span>· Color: <strong>${escHtml(color)}</strong></span>` : ''}
              ${shape ? `<span>· Shape: <strong>${escHtml(shape)}</strong></span>` : ''}
              ${cat   ? `<span>· Category: <strong>${escHtml(cat)}</strong></span>` : ''}
            </div>
            <div class="d-flex gap-1 flex-wrap">
              <button class="btn btn-success btn-sm"
                      data-confirm-btn data-name="${escHtml(r.medication)}">
                ✓ Confirm
              </button>
              ${r.drug_id
                ? `<a href="/medication/${r.drug_id}" class="btn btn-outline btn-sm"><i class="fa-solid fa-circle-info"></i> View Details</a>`
                : ''}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderSuggestion(r) {
    const conf  = r.confidence_pct || Math.round(r.confidence_score * 100);
    const level = r.confidence_level || (conf >= 80 ? 'high' : conf >= 60 ? 'medium' : 'low');
    const imgHtml = r.image_url
      ? `<div style="width:100%;height:72px;border-radius:var(--radius-sm);overflow:hidden;
                     border:1px solid var(--border);background:var(--bg);margin-bottom:8px;
                     display:flex;align-items:center;justify-content:center">
           <img src="${escHtml(r.image_url)}" alt="${escHtml(r.medication)}"
                style="max-width:100%;max-height:100%;object-fit:contain;padding:4px"
                onerror="this.parentElement.style.display='none'">
         </div>`
      : '';
    return `
      <div class="suggestion-card" ${r.drug_id ? `style="cursor:pointer" onclick="window.location='/medication/${r.drug_id}'"` : ''}>
        ${imgHtml}
        <div class="suggestion-name">${escHtml(r.medication)}</div>
        <div class="suggestion-generic">${escHtml(r.generic_name || '')}</div>
        <div class="mt-1" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:4px">
          <span class="confidence-badge ${level}" style="font-size:.7rem">${conf}%</span>
          ${r.drug_id ? `<a href="/medication/${r.drug_id}" class="btn btn-outline btn-sm" style="font-size:.7rem;padding:2px 8px" onclick="event.stopPropagation()">Details</a>` : ''}
        </div>
      </div>
    `;
  }

  function renderNoMatch() {
    return `
      <div class="no-result-box">
        <div class="no-result-icon">🔍</div>
        <div class="no-result-title">No confident match found.</div>
        <div class="no-result-text">
          CAPSULE could not find a reliable medication match for this description.
        </div>
        <ul style="text-align:left;margin:16px auto;max-width:320px;font-size:.85rem;color:var(--text-secondary)">
          <li>Check the spelling in the description.</li>
          <li>Add more details: color, shape, packaging, imprint, or intended use.</li>
          <li>Try searching manually using other clinical systems.</li>
        </ul>
      </div>
    `;
  }

  function renderIncomplete(comp) {
    const missing = (comp && comp.missing) ? comp.missing : [];
    return `
      <div class="no-result-box">
        <div class="no-result-icon">⚠️</div>
        <div class="no-result-title">The description is incomplete.</div>
        <div class="no-result-text">
          CAPSULE needs more information before it can identify a medication.
        </div>
        ${missing.length ? `
          <p style="margin-top:12px;font-size:.85rem;font-weight:600">Please add:</p>
          <ul style="text-align:left;margin:0 auto;max-width:320px;font-size:.85rem;color:var(--text-secondary)">
            ${missing.map(m => `<li>${escHtml(m)}</li>`).join('')}
          </ul>
        ` : ''}
      </div>
    `;
  }

  function setLoading(on, el) {
    if (!el) return;
    el.style.display = on ? 'flex' : 'none';
  }
})();

// ============================================================
// PHARMACIST — Drug Database Search
// ============================================================
(function initPharmacistSearch() {
  const searchInput = document.getElementById('drugSearchInput');
  const categoryFilter = document.getElementById('categoryFilter');
  const searchBtn = document.getElementById('drugSearchBtn');
  const drugContainer = document.getElementById('drugContainer');
  if (!searchInput || !drugContainer) return;

  let debounceTimer;

  async function fetchDrugs() {
    const q = searchInput.value.trim();
    const cat = categoryFilter ? categoryFilter.value : '';

    // Show loading state
    drugContainer.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-secondary)"><i class="fa-solid fa-spinner fa-spin" style="font-size:1.5rem"></i><div style="margin-top:8px">Searching…</div></div>';

    try {
      const url = `/api/drugs/search?q=${encodeURIComponent(q)}&category=${encodeURIComponent(cat)}&per_page=50`;
      const resp = await fetch(url);
      const data = await resp.json();
      renderDrugCards(data.drugs, drugContainer);
      const countEl = document.getElementById('drugCount');
      if (countEl) countEl.textContent = `${data.total} medicines`;
    } catch (err) {
      console.error(err);
      drugContainer.innerHTML = '<div class="no-result-box"><div class="no-result-title">Search failed. Please try again.</div></div>';
    }
  }

  function fetchDrugsDebounced() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fetchDrugs, 300);
  }

  // Search on typing (debounced)
  searchInput.addEventListener('input', fetchDrugsDebounced);

  // Search on Enter key
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      clearTimeout(debounceTimer);
      fetchDrugs();
    }
  });

  // Search button click
  if (searchBtn) searchBtn.addEventListener('click', function() {
    clearTimeout(debounceTimer);
    fetchDrugs();
  });

  // Category filter — search immediately on change
  if (categoryFilter) categoryFilter.addEventListener('change', function() {
    clearTimeout(debounceTimer);
    fetchDrugs();
  });

  function renderDrugCards(drugs, container) {
    if (!drugs.length) {
      const q = searchInput.value.trim();
      const cat = categoryFilter ? categoryFilter.value : '';
      let msg = 'No medicines found.';
      if (q && cat) msg = `No medicines found for "${q}" in category "${cat}".`;
      else if (q) msg = `No medicines found for "${q}".`;
      else if (cat) msg = `No medicines found in category "${cat}".`;
      container.innerHTML = `<div class="no-result-box"><div class="no-result-title">${msg}</div><div style="font-size:.85rem;color:var(--text-secondary);margin-top:8px">Try a different search term or category.</div></div>`;
      return;
    }
    container.innerHTML = drugs.map(d => `
      <div class="drug-card" onclick="window.location='/medication/${d.id}'">
        <div class="drug-card-name">${escHtml(d.drug_name)}${d.drug_name_ar ? ` <span style="font-weight:400;color:var(--text-secondary);font-size:.82rem">${escHtml(d.drug_name_ar)}</span>` : ''}</div>
        <div class="drug-card-generic">${escHtml(d.generic_name)}${d.generic_name_ar ? ` — ${escHtml(d.generic_name_ar)}` : ''}</div>
        <div class="drug-card-tags">
          ${d.dosage_form ? `<span class="drug-tag">${escHtml(d.dosage_form)}</span>` : ''}
          ${d.strength ? `<span class="drug-tag">${escHtml(d.strength)}</span>` : ''}
          ${d.category ? `<span class="drug-tag">${escHtml(d.category)}</span>` : ''}
        </div>
      </div>
    `).join('');
  }
})();

// ============================================================
// ADMIN — User role select
// ============================================================
document.querySelectorAll('.role-select-form select').forEach(sel => {
  sel.addEventListener('change', function () {
    this.closest('form').submit();
  });
});

// ============================================================
// STATS — Live counter animation
// ============================================================
function animateCounter(el) {
  const target = parseInt(el.dataset.target || el.textContent, 10);
  if (isNaN(target)) return;
  let current = 0;
  const step = Math.max(1, Math.round(target / 40));
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = current.toLocaleString();
    if (current >= target) clearInterval(timer);
  }, 30);
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(entry.target);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: .3 });

document.querySelectorAll('.stat-value[data-target]').forEach(el => observer.observe(el));

// ============================================================
// UTILITIES
// ============================================================
function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showAlert(container, msg, type) {
  if (!container) return;
  container.innerHTML = `
    <div class="alert alert-${type}" data-auto-dismiss>
      <span>${escHtml(msg)}</span>
    </div>
  `;
}

function showToast(msg, type = 'success') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    padding:12px 20px; border-radius:10px; font-size:.875rem;
    font-weight:600; box-shadow:0 4px 12px rgba(0,0,0,.15);
    background:${type === 'success' ? '#d1fae5' : '#fee2e2'};
    color:${type === 'success' ? '#065f46' : '#991b1b'};
    animation:slideIn .3s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 2800);
}

// Add slide-in animation
const style = document.createElement('style');
style.textContent = '@keyframes slideIn { from { transform:translateY(20px); opacity:0; } to { transform:none; opacity:1; } }';
document.head.appendChild(style);

// Add medication directly from Medication Detail page
document.addEventListener('DOMContentLoaded', () => {
  const addDetailBtn = document.querySelector('.add-detail-med-btn');
  const msg = document.getElementById('addMedicationMsg');

  if (!addDetailBtn) return;

  addDetailBtn.addEventListener('click', async () => {
    const medicationName = addDetailBtn.dataset.name || '';
    const drugId = addDetailBtn.dataset.drugId || '';

    addDetailBtn.disabled = true;
    addDetailBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Adding...';

    try {
      const response = await fetch('/api/patient/confirm-medication', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          medication_name: medicationName,
          drug_id: drugId
        })
      });

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.message || 'Could not add medication.');
      }

      addDetailBtn.innerHTML = '<i class="fa-solid fa-check"></i> Added to My Medications';
      addDetailBtn.style.background = 'var(--success)';

      if (msg) {
        msg.style.color = '#047857';
        msg.textContent = data.message || 'added successfully.';
      }

    } catch (error) {
      addDetailBtn.disabled = false;
      addDetailBtn.innerHTML = '<i class="fa-solid fa-plus"></i> Add to My Medications';

      if (msg) {
        msg.style.color = '#b91c1c';
        msg.textContent = error.message || 'Failed to add medication.';
      }
    }
  });
});
