/* timeline.js — chronological journal with date grouping and scroll animation */
import { navigate } from './app.js';
import { t } from './i18n.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';

let _filter = 'all';

export async function init(container) {
  const [galRes, msRes] = await Promise.all([
    fetch('data/gallery.json'), fetch('data/milestones.json')
  ]);
  const gallery = await galRes.json();
  const milestones = await msRes.json();

  const allNodes = [
    ...gallery.map(g => ({ type: 'image', date: g.date, data: g })),
    ...milestones.map(m => ({ type: 'milestone', date: m.date, data: m }))
  ].sort((a, b) => b.date.localeCompare(a.date));

  buildFilters(container, allNodes);
  renderTimeline(container, allNodes);
}

function buildFilters(container, nodes) {
  const bar = document.createElement('div');
  bar.className = 'timeline-filters';
  ['all', 'image', 'milestone'].forEach(key => {
    const btn = document.createElement('button');
    btn.className = 'timeline-filter' + (key === 'all' ? ' active' : '');
    btn.textContent = key === 'all' ? t('gallery.all') : key === 'image' ? t('nav.gallery') : t('nav.timeline');
    btn.addEventListener('click', () => {
      _filter = key;
      bar.querySelectorAll('.timeline-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tl = container.querySelector('.timeline');
      if (tl) tl.remove();
      renderTimeline(container, nodes);
    });
    bar.appendChild(btn);
  });
  container.appendChild(bar);
}

function renderTimeline(container, allNodes) {
  const nodes = allNodes.filter(n => _filter === 'all' || n.type === _filter);
  const groups = {};
  nodes.forEach(n => { (groups[n.date] ??= []).push(n); });

  const tl = document.createElement('div');
  tl.className = 'timeline';

  Object.keys(groups).sort((a, b) => b.localeCompare(a)).forEach(date => {
    const header = document.createElement('div');
    header.className = 'timeline-date-header';
    header.textContent = date;
    tl.appendChild(header);

    groups[date].forEach(node => {
      const el = document.createElement('div');
      const isMilestone = node.type === 'milestone';
      el.className = 'timeline-node' + (isMilestone ? ' milestone' : ' has-image');

      if (isMilestone) {
        const m = node.data;
        el.innerHTML = `
          <div class="timeline-title">${m.title}</div>
          <div class="timeline-desc">${m.description}</div>
          <a class="timeline-commit" href="https://github.com/teenu/4cgt/tree/${m.commit}" target="_blank" rel="noopener">${m.commit}</a>
        `;
      } else {
        const g = node.data;
        el.innerHTML = `
          <div class="timeline-title">${g.character || g.tags[0] || g.id}${g.series ? ' — ' + g.series : ''}</div>
          <img class="timeline-thumb" src="${HF}${g.image}" alt="${g.character || g.id}" loading="lazy" />
        `;
        el.querySelector('.timeline-thumb').addEventListener('click', () => navigate('#/gallery/' + g.id));
      }
      tl.appendChild(el);
    });
  });

  container.appendChild(tl);
  observeNodes(tl);
}

function observeNodes(tl) {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });
  tl.querySelectorAll('.timeline-node').forEach(n => obs.observe(n));
}
