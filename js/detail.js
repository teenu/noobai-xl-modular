/* detail.js — single image detail view */
import { navigate } from './app.js';
import { t } from './i18n.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';

export function showDetail(id, gallery) {
  const g = gallery.find(e => e.id === id);
  if (!g) { navigate('#/'); return; }

  const view = document.getElementById('view-detail');
  view.classList.add('active');

  const idx = gallery.indexOf(g);
  const prev = idx > 0 ? gallery[idx - 1] : null;
  const next = idx < gallery.length - 1 ? gallery[idx + 1] : null;

  view.innerHTML = `
    <a class="detail-back" id="detail-back">&larr; ${t('nav.gallery')}</a>
    <div class="detail-frame">
      <img src="${HF}${g.image}" alt="${g.character || g.id}" />
    </div>
    <div class="detail-meta">
      <div class="detail-info">
        <h2>${g.character || g.tags[0] || g.id}</h2>
        ${g.series ? `<div class="series">${g.series}</div>` : ''}
        <div class="date">${g.date}</div>
        ${g.commentary ? `<div class="detail-commentary">${g.commentary}</div>` : ''}
        <a class="detail-commit" href="https://github.com/teenu/4cgt/tree/${g.commit}" target="_blank" rel="noopener">${t('gallery.commit')}: ${g.commit}</a>
        <div class="detail-reproduce" style="margin-top:16px">
          <a href="https://github.com/teenu/4cgt" class="btn btn-primary" style="padding:8px 20px;font-size:13px" target="_blank" rel="noopener">${t('gallery.reproduce')}</a>
        </div>
      </div>
      <div class="detail-settings">
        <table>
          <tr><td>${t('gallery.seed')}</td><td>${g.settings.seed}</td></tr>
          <tr><td>${t('gallery.steps')}</td><td>${g.settings.steps}</td></tr>
          <tr><td>${t('gallery.cfg')}</td><td>${g.settings.cfg_scale}</td></tr>
          <tr><td>${t('gallery.rescale')}</td><td>${g.settings.rescale_cfg}</td></tr>
          <tr><td>${t('gallery.resolution')}</td><td>${g.settings.width}x${g.settings.height}</td></tr>
          <tr><td>${t('gallery.model')}</td><td>${g.settings.model}</td></tr>
          <tr><td>${t('gallery.scheduler')}</td><td>${g.settings.scheduler}</td></tr>
          ${g.dora.enabled ? `
          <tr><td>${t('gallery.dora')}</td><td>${g.dora.adapter}</td></tr>
          <tr><td>${t('gallery.mode')}</td><td>${g.dora.mode}</td></tr>
          <tr><td>${t('gallery.strength')}</td><td>${g.dora.strength}</td></tr>
          <tr><td>${t('gallery.startStep')}</td><td>${g.dora.start_step}</td></tr>
          ` : ''}
        </table>
        <details class="detail-prompt">
          <summary>${t('gallery.prompt')}</summary>
          <pre>${g.prompt}</pre>
        </details>
        <details class="detail-prompt">
          <summary>${t('gallery.negPrompt')}</summary>
          <pre>${g.negative_prompt}</pre>
        </details>
      </div>
    </div>
    <div class="detail-nav">
      <button id="detail-prev" ${!prev ? 'disabled' : ''}>&larr; ${t('gallery.prev')}</button>
      <button id="detail-next" ${!next ? 'disabled' : ''}>${t('gallery.next')} &rarr;</button>
    </div>
    <div class="detail-support">
      <a href="#/support">${t('gallery.supportLine')}</a>
    </div>
  `;

  view.querySelector('#detail-back').addEventListener('click', e => { e.preventDefault(); navigate('#/'); });
  if (prev) view.querySelector('#detail-prev').addEventListener('click', () => navigate('#/gallery/' + prev.id));
  if (next) view.querySelector('#detail-next').addEventListener('click', () => navigate('#/gallery/' + next.id));
  window.scrollTo(0, 0);
}

/* Keyboard navigation for detail view */
export function initKeyboard() {
  document.addEventListener('keydown', e => {
    const view = document.getElementById('view-detail');
    if (!view.classList.contains('active')) return;
    if (e.key === 'ArrowLeft') { const btn = view.querySelector('#detail-prev'); if (btn && !btn.disabled) btn.click(); }
    if (e.key === 'ArrowRight') { const btn = view.querySelector('#detail-next'); if (btn && !btn.disabled) btn.click(); }
    if (e.key === 'Escape') navigate('#/');
  });
}
