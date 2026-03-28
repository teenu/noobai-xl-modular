/* app.js — SPA router, age gate, hero, init */
import * as i18n from './i18n.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';
const HERO_IMAGES = [
  'frieren_beach.png','lucy_cyberpunk.png','miku.png','makima.png',
  'asuka.png','2b_ruins.png','zero_two.png','yor.png'
];

/* Router */
export function navigate(hash) {
  window.location.hash = hash;
}

function getRoute() {
  const h = window.location.hash.slice(1) || '/';
  return h.startsWith('/') ? h : '/' + h;
}

let _onRoute = null;
export function onRoute(cb) { _onRoute = cb; }

function handleRoute() {
  const route = getRoute();
  // Hide all views
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  // Update nav active state
  document.querySelectorAll('.nav-links a[data-route]').forEach(a => {
    a.classList.toggle('active', a.dataset.route === route || route.startsWith(a.dataset.route + '/'));
  });
  if (_onRoute) _onRoute(route);
}

/* Age gate */
function initAgeGate() {
  const gate = document.getElementById('age-gate');
  if (localStorage.getItem('4cgt_age_verified') === 'true') {
    gate.classList.remove('active');
    return;
  }
  document.getElementById('age-confirm').addEventListener('click', () => {
    localStorage.setItem('4cgt_age_verified', 'true');
    gate.classList.remove('active');
  });
  document.getElementById('age-deny').addEventListener('click', () => {
    window.location.href = 'https://www.google.com';
  });
}

/* Hero slideshow */
function initHero() {
  const bg = document.getElementById('hero-bg');
  HERO_IMAGES.forEach((img, i) => {
    const div = document.createElement('div');
    div.className = 'hero-slide' + (i === 0 ? ' active' : '');
    div.style.backgroundImage = `url('${HF}${img}')`;
    bg.appendChild(div);
  });
  const slides = bg.querySelectorAll('.hero-slide');
  let idx = 0;
  setInterval(() => {
    slides[idx].classList.remove('active');
    idx = (idx + 1) % slides.length;
    slides[idx].classList.add('active');
  }, 6000);

  const hint = document.querySelector('.scroll-hint');
  window.addEventListener('scroll', () => {
    hint.classList.toggle('fade', window.scrollY > 80);
  }, { passive: true });
}

/* Nav burger */
function initNav() {
  const burger = document.querySelector('.nav-burger');
  const links = document.querySelector('.nav-links');
  burger.addEventListener('click', () => links.classList.toggle('open'));
  links.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => links.classList.remove('open'));
  });

  // Language switcher
  const langBtn = document.querySelector('.nav-lang');
  const langs = ['en','ja','zh','fr'];
  const labels = { en:'EN', ja:'\u65E5\u672C\u8A9E', zh:'\u4E2D\u6587', fr:'FR' };
  langBtn.textContent = labels[i18n.currentLang()] || 'EN';
  langBtn.addEventListener('click', () => {
    const cur = langs.indexOf(i18n.currentLang());
    const next = langs[(cur + 1) % langs.length];
    i18n.setLang(next);
    langBtn.textContent = labels[next];
  });
}

/* Init */
export async function init() {
  await i18n.init();
  initAgeGate();
  initHero();
  initNav();
  window.addEventListener('hashchange', handleRoute);
  handleRoute();
}
