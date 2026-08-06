const API_URL = "http://127.0.0.1:5000/analyze";

const $ = (id) => document.getElementById(id);

chrome.storage.local.get(["apiKey", "provider", "draftQuestion", "draftSource"], (data) => {
    if (data.apiKey) $("apiKey").value = data.apiKey;
    if (data.provider) $("provider").value = data.provider;
    if (data.draftQuestion) $("question").value = data.draftQuestion;
    if (data.draftSource) $("source").value = data.draftSource;
});

$("question").addEventListener("input", () => {
    chrome.storage.local.set({ draftQuestion: $("question").value });
});
$("source").addEventListener("input", () => {
    chrome.storage.local.set({ draftSource: $("source").value });
});

chrome.storage.local.get(["pendingPick"], (data) => {
    if (!data.pendingPick) return;
    const { target, text } = data.pendingPick;
    if (target === "question") {
        $("question").value = text;
        chrome.storage.local.set({ draftQuestion: text });
    }
    if (target === "source") {
        $("source").value = text;
        chrome.storage.local.set({ draftSource: text });
    }
    chrome.storage.local.remove("pendingPick");
    setStatus(`Filled ${target} from your page selection.`);
});

async function startPicking(target) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.url || !/^https?:\/\//.test(tab.url)) {
        setStatus("Can't pick on this page — open a normal http(s) webpage first.");
        return;
    }

    let hadError = false;
    await new Promise((resolve) => {
        chrome.tabs.sendMessage(tab.id, { type: "START_PICKING", target }, () => {
            if (chrome.runtime.lastError) {
                hadError = true;
                setStatus("No content script on this page — try reloading the tab.");
            }
            resolve();
        });
    });

    if (!hadError) window.close();
}

$("pickQuestion").addEventListener("click", () => startPicking("question"));
$("pickSource").addEventListener("click", () => startPicking("source"));

$("analyze").addEventListener("click", async () => {
    const question = $("question").value.trim();
    const source_snippet = $("source").value.trim();
    const provider = $("provider").value;
    const api_key = $("apiKey").value.trim();

    if (!question) {
        setStatus("Enter a question first.");
        return;
    }

    chrome.storage.local.set({ apiKey: api_key, provider });

    setStatus("Analyzing…");
    $("result").classList.add("hidden");
    $("analyze").disabled = true;

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, source_snippet, provider, api_key }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || `Server returned ${res.status}`);
        }

        const data = await res.json();
        renderResult(data);
        setStatus("");
        chrome.storage.local.remove(["draftQuestion", "draftSource"]);

        if ($("showOnPage").checked) {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            chrome.tabs.sendMessage(tab.id, { type: "SHOW_RESULT_ON_PAGE", data }, () => {
                void chrome.runtime.lastError;
            });
        }
    } catch (e) {
        setStatus(
            e.message.includes("Failed to fetch")
                ? "Can't reach the local server. Is `python backend/api.py` running?"
                : e.message
        );
    } finally {
        $("analyze").disabled = false;
    }
});

function renderResult(data) {
    const tagEl = $("tag");
    tagEl.textContent = data.tag;
    tagEl.className = "tag " + data.tag.toLowerCase().replace(/\s+/g, "-");

    $("answer").textContent = data.answer;
    $("perplexity").textContent = `PPL: ${data.perplexity.toFixed(1)}`;
    $("entailment").textContent = `Entail: ${(data.entailment * 100).toFixed(0)}%`;
    $("contradiction").textContent = `Contra: ${(data.contradiction * 100).toFixed(0)}%`;
    $("rationale").textContent = data.rationale;

    $("result").classList.remove("hidden");
}

function setStatus(msg) {
    $("status").textContent = msg;
}