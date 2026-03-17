<script type="module">
const SPEAK_DELAY = 2000;
const root = (window.parent && window.parent.window) ? window.parent.window : window;
const doc = (root.document) ? root.document : document;

let timer = null;
let activeElem = null;
let lastSpoken = "";

function cancelTimer() {
  if (timer) clearTimeout(timer);
  timer = null;
  if (activeElem) {
    try {
      activeElem.style.outline = "";
    } catch (error) {
      // ignore style cleanup errors
    }
    activeElem = null;
  }
}

function speakButtonText(elem) {
  const text = elem.dataset.tts || elem.innerText || elem.textContent || "";
  if (!text || text === lastSpoken) return;

  lastSpoken = text;
  try {
    elem.style.outline = "2px solid orange";
    setTimeout(() => {
      elem.style.outline = "";
    }, 900);
  } catch (error) {
    // ignore style update errors
  }

  fetch("http://localhost:__TTS_PORT__/_tts?text=" + encodeURIComponent(text)).catch(() => {});
}

function startTimer(elem) {
  cancelTimer();
  activeElem = elem;
  timer = setTimeout(() => speakButtonText(elem), SPEAK_DELAY);
}

function installHoverListeners() {
  if (root.__ttsHoverInstalled) return;
  root.__ttsHoverInstalled = true;

  doc.addEventListener("mouseover", (event) => {
    const btn = event.target.closest("button");
    if (btn && btn.textContent.trim() !== "") startTimer(btn);
  });

  doc.addEventListener("mouseout", (event) => {
    if (event.target.closest("button")) cancelTimer();
  });

  doc.addEventListener(
    "touchstart",
    (event) => {
      const btn = event.target.closest("button");
      if (btn && btn.textContent.trim() !== "") startTimer(btn);
    },
    { passive: true },
  );

  doc.addEventListener("touchend", () => cancelTimer(), { passive: true });
}

function applyMappings(metaTexts, optTexts) {
  try {
    const allButtons = doc.querySelectorAll("button, [role=button]");
    let metaIndex = 0;
    let optIndex = 0;

    allButtons.forEach((button) => {
      const label = (button.innerText || button.textContent || "").trim();

      if (["🆘", "😕", "👍", "👎"].includes(label)) {
        if (metaTexts[metaIndex]) {
          button.dataset.tts = metaTexts[metaIndex];
        }
        metaIndex += 1;
        return;
      }

      if (optTexts[optIndex]) {
        button.dataset.tts = optTexts[optIndex];
      }
      optIndex += 1;
    });
  } catch (error) {
    setTimeout(() => applyMappings(metaTexts, optTexts), 400);
  }
}

function main() {
  installHoverListeners();
  const metaTexts = __META__;
  const optTexts = __OPTS__;
  applyMappings(metaTexts, optTexts);
}

main();
</script>
