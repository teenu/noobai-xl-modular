/* timeline.js — chronological journal of gallery + milestones */
import { navigate } from './app.js';
import { t } from './i18n.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';

export async function init(container) {
  const [galRes, msRes] = await Promise.all([
    fetch('data/gallery.json'), fetch('data/milestones.json')
  ]);
  const gallery = await galRes.json();
  const milestones = await msRes.json();

  // Merge and sort by date descending
  const nodes = [
    ...gallery.map(g => ({ type: 'image', date: g.date, data: g })),
    ...milestones.map(m => ({ type: 'milestone', date: m.date, data: m }))
  ].sort((a, b) => b.date.localeCompare(a.date));

  const tl = document.createElement('div');
  tl.className = 'timeline';

  nodes.forEach(node => {
    const el = document.createElement('div');
    el.className = 'timeline-node' + (node.type === 'image' ? ' has-image' : '');

    if (node.type === 'image') {
      const g = node.data;
      el.innerHTML = `
        <div class="timeline-date">${g.date}
          <a class="timeline-commit" href="https://github.com/teenu/4cgt/tree/${g.commit}" target="_blank" rel="noopener">${g.commit}</a>
        </div>
        <div class="timeline-title">${g.character || g.tags[0] || g.id}${g.series ? ' — ' + g.series : ''}</div>
        <img class="timeline-thumb" src="${HF}${g.image}" alt="${g.character || g.id}" loading="lazy" />
      `;
      el.querySelector('.timeline-thumb').addEventListener('click', () => navigate('#/gallery/' + g.id));
    } else {
      const m = node.data;
      el.innerHTML = `
        <div class="timeline-date">${m.date}
          <a class="timeline-commit" href="https://github.com/teenu/4cgt/tree/${m.commit}" target="_blank" rel="noopener">${m.commit}</a>
        </div>
        <div class="timeline-title">${m.title}</div>
        <div class="timeline-desc">${m.description}</div>
      `;
    }
    tl.appendChild(el);
  });

  container.appendChild(tl);
}
