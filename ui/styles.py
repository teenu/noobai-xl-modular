"""UI styles and scripts for Gradio interface."""

CSS_STYLES = """
.title-text {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}
.status-success {
    background: rgba(34, 197, 94, 0.1) !important;
    color: rgb(34, 197, 94) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
}
.status-error {
    background: rgba(239, 68, 68, 0.1) !important;
    color: rgb(239, 68, 68) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}
.segment-container {
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
}
.segment-header {
    font-weight: bold;
    margin-bottom: 8px;
}
.final-prompt-container {
    background: var(--background-fill-secondary);
    border: 2px solid var(--border-color-accent);
    border-radius: 8px;
    padding: 15px;
}
.dora-grid-container {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    padding: 10px;
    background: var(--background-fill-secondary);
    border-radius: 6px;
    margin-top: 10px;
}
.dora-cell {
    width: 18px;
    height: 18px;
    background: #dc2626 !important;
    border-radius: 2px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.dora-cell.on {
    background: #16a34a !important;
}
.dora-cell:hover {
    transform: scale(1.1);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.dora-preset-badge {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.8em;
    margin-bottom: 8px;
    display: inline-block;
}
"""

JAVASCRIPT_HEAD = """
<script>
function setupDoraGridHandlers() {
    setTimeout(function() {
        const containers = document.querySelectorAll('[id*="dora-grid"][id$="-container"]');

        containers.forEach(function(container) {
            if (container.hasAttribute('data-dora-initialized')) return;
            container.setAttribute('data-dora-initialized', 'true');

            container.addEventListener('click', function(e) {
                const cell = e.target;
                if (!cell.classList.contains('dora-cell')) return;

                cell.classList.toggle('on');

                const cells = container.querySelectorAll('.dora-cell');
                const schedule = Array.from(cells).map(c =>
                    c.classList.contains('on') ? '1' : '0'
                );
                const scheduleCSV = schedule.join(', ');

                const hiddenBox = document.getElementById('dora_manual_schedule_hidden');
                if (hiddenBox) {
                    const input = hiddenBox.querySelector('textarea, input');
                    if (input) {
                        input.value = scheduleCSV;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }
            });
        });
    }, 500);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupDoraGridHandlers);
} else {
    setupDoraGridHandlers();
}

const observer = new MutationObserver(setupDoraGridHandlers);
setTimeout(function() {
    observer.observe(document.body, { childList: true, subtree: true });
}, 1000);
</script>
"""
