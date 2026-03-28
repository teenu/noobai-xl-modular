/* gallery.js — grid rendering, filtering */
import { navigate } from './app.js';
import { t } from './i18n.js';
import { showDetail, initKeyboard } from './detail.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';
const IS_TOUCH = matchMedia('(hover: none)').matches;
let _gallery = [];
let _filter = 'all';
let _search = '';

export async function init() {
  const res = await fetch('data/gallery.json');
  _gallery = await res.json();
  buildFilterBar();
  renderGrid();
  initKeyboard();
}

export function route(path) {
  if (path.startsWith('/gallery/')) {
    showDetail(path.slice(9), _gallery);
    return true;
  }
  if (path === '/' || path === '/gallery') {
    document.getElementById('view-home').classList.add('active');
    renderGrid();
    return true;
  }
  return false;
}

/* Filter bar */
function buildFilterBar() {
  const bar = document.getElementById('filter-bar');
  bar.innerHTML = '';
  const series = [...new Set(_gallery.map(g => g.series).filter(Boolean))];

  addFilterBtn(bar, 'all', t('gallery.all'));
  series.forEach(s => addFilterBtn(bar, s, s));
  addFilterBtn(bar, '__originals', t('gallery.originals'));

  const input = document.createElement('input');
  input.className = 'filter-search';
  input.type = 'text';
  input.setAttribute('data-i18n', 'gallery.search');
  input.placeholder = t('gallery.search');
  input.addEventListener('input', e => { _search = e.target.value.toLowerCase(); renderGrid(); });
  bar.appendChild(input);
}

function addFilterBtn(bar, key, label) {
  const btn = document.createElement('button');
  btn.className = 'filter-btn' + (key === _filter ? ' active' : '');
  btn.textContent = label;
  btn.addEventListener('click', () => {
    _filter = key;
    bar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderGrid();
  });
  bar.appendChild(btn);
}

/* Grid */
function renderGrid() {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = '';
  const filtered = _gallery.filter(g => {
    if (_filter === '__originals') return !g.series;
    if (_filter !== 'all' && g.series !== _filter) return false;
    if (_search) {
      const hay = [g.character, g.series, ...g.tags, g.id].filter(Boolean).join(' ').toLowerCase();
      if (!hay.includes(_search)) return false;
    }
    return true;
  });

  filtered.forEach((g, i) => {
    const fig = document.createElement('figure');
    fig.className = 'gallery-item';
    fig.style.setProperty('--stagger', i);
    const img = document.createElement('img');
    img.src = HF + g.image;
    img.alt = (g.character || g.id) + (g.series ? ' — ' + g.series : '');
    img.loading = 'lazy';
    img.decoding = 'async';
    img.onload = () => fig.classList.add('loaded');
    const cap = document.createElement('figcaption');
    cap.textContent = g.character || g.tags[0] || g.id;
    fig.appendChild(img);
    fig.appendChild(cap);
    fig.addEventListener('click', () => navigate('#/gallery/' + g.id));
    if (!IS_TOUCH) addTilt(fig);
    grid.appendChild(fig);
  });
}

/* 3D perspective tilt — desktop only */
function addTilt(el) {
  el.addEventListener('mousemove', e => {
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    el.style.transform = `perspective(800px) rotateY(${x * 10}deg) rotateX(${-y * 10}deg) scale(1.03)`;
  });
  el.addEventListener('mouseleave', () => { el.style.transform = ''; });
}
