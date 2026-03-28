/* i18n — translation engine for 4CGT */
let _strings = {};
let _lang = 'en';

export function currentLang() { return _lang; }

export async function init() {
  const stored = localStorage.getItem('4cgt_lang');
  const browser = (navigator.language || '').slice(0, 2);
  _lang = stored || (['ja','zh','fr'].includes(browser) ? browser : 'en');
  try {
    const res = await fetch('data/i18n.json');
    _strings = await res.json();
  } catch { _strings = {}; }
  apply();
}

export function setLang(lang) {
  _lang = lang;
  localStorage.setItem('4cgt_lang', lang);
  apply();
}

export function t(key) {
  return (_strings[_lang] && _strings[_lang][key]) ||
         (_strings.en && _strings.en[key]) || key;
}

function apply() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    const val = t(key);
    if (el.placeholder !== undefined && el.tagName === 'INPUT') el.placeholder = val;
    else el.textContent = val;
  });
  document.documentElement.lang = _lang;
}
