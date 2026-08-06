// content.js — runs on every page.
//   1. Responds to a request from the popup asking for whatever text the
//      user currently has selected, so it can be used as a "source snippet"
//      without retyping it.
//   2. Injects the labeler's result as a small badge directly into the page,
//      right after the text the user had selected (e.g. under an AI chat
//      response), so the tag lives next to the content it's judging instead
//      of only living in the popup.

let lastSelectionRange = null;
let pickingTarget = null; // "question" | "source" | null

// Keep track of the most recent selection's Range object (not just its
// text) so we know *where* on the page to inject the result badge later.
document.addEventListener("selectionchange", () => {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0 && sel.toString().trim()) {
        lastSelectionRange = sel.getRangeAt(0).cloneRange();
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "GET_SELECTION") {
        sendResponse({ selection: window.getSelection().toString() });
    }

    if (message.type === "SHOW_RESULT_ON_PAGE") {
        injectResultBadge(message.data);
        sendResponse({ ok: true });
    }

    if (message.type === "START_PICKING") {
        startPicking(message.target);
        sendResponse({ ok: true });
    }

    return true; // keep the message channel open for the async sendResponse
});

function startPicking(target) {
    pickingTarget = target;
    document.body.classList.add("ahcl-picking");
    showPickHint(target);
    document.addEventListener("mouseup", onPickMouseUp, { once: true });
    // Let Escape cancel picking without selecting anything.
    document.addEventListener("keydown", onPickEscape);
}

function onPickEscape(e) {
    if (e.key === "Escape") stopPicking();
}

function stopPicking() {
    pickingTarget = null;
    document.body.classList.remove("ahcl-picking");
    document.removeEventListener("keydown", onPickEscape);
    removePickHint();
}

function onPickMouseUp() {
    const text = window.getSelection().toString().trim();
    const target = pickingTarget;
    stopPicking();

    if (!text || !target) return;

    chrome.storage.local.set({
        pendingPick: { target, text, ts: Date.now() },
    });
    showPickToast(target);
}

function showPickHint(target) {
    removePickHint();
    const hint = document.createElement("div");
    hint.id = "ahcl-pick-hint";
    hint.textContent = `Select text for "${target}" — drag over it now (Esc to cancel)`;
    document.body.appendChild(hint);
}

function removePickHint() {
    document.getElementById("ahcl-pick-hint")?.remove();
}

function showPickToast(target) {
    const toast = document.createElement("div");
    toast.id = "ahcl-pick-toast";
    toast.textContent = `Copied to ${target === "question" ? "Question" : "Source"} — reopen the extension to see it`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function injectResultBadge(data) {
    const badge = buildBadge(data);

    if (lastSelectionRange) {
        try {
            // Insert right after the end of the selected text, so it visually
            // sits directly under/beside whatever the user selected.
            const range = lastSelectionRange.cloneRange();
            range.collapse(false); // move to the end of the selection
            range.insertNode(badge);
            return;
        } catch (e) {
            // Selection may have gone stale (e.g. page re-rendered) — fall through
            // to the floating fallback below instead of failing silently.
        }
    }

    // Fallback: no valid selection anchor — show as a floating badge in the
    // corner instead of failing silently.
    badge.classList.add("ahcl-floating");
    document.body.appendChild(badge);
    setTimeout(() => badge.remove(), 15000);
}

function buildBadge(data) {
    const el = document.createElement("div");
    el.className = "ahcl-badge ahcl-" + data.tag.toLowerCase().replace(/\s+/g, "-");
    el.innerHTML = `
    <strong>${escapeHtml(data.tag)}</strong>
    <span class="ahcl-close">✕</span>
    <div class="ahcl-detail">${escapeHtml(data.rationale)}</div>
  `;
    el.querySelector(".ahcl-close").addEventListener("click", (e) => {
        e.stopPropagation();
        el.remove();
    });
    return el;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}