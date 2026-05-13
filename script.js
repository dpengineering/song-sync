let lastPressTime = null;
let startTime = null;
let isRunning = true;
const logDiv = document.getElementById("log");
const exportBtn = document.getElementById("export");
const copyBtn = document.getElementById("copyButton");
const customTextInput = document.getElementById("customText");
const customDelayNameInput = document.getElementById("customDelayName");
const arduinoCodeDiv = document.getElementById("arduinoCode");
const enableOffsetCheckbox = document.getElementById("enableOffset");
const offsetSignSelect = document.getElementById("offsetSign");
const cumulativeTimingCheckbox = document.getElementById("cumulativeTiming");
const themeToggleBtn = document.getElementById("themeToggle");
let log = [];
let delays = [];
let totalElapsed = 0;
let holdStart = null;
let lastPulseEnd = null;
let actions = [];

const preferredTheme = localStorage.getItem("song-sync-theme")
  || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

const applyTheme = (theme) => {
  document.body.classList.toggle("dark", theme === "dark");
  themeToggleBtn.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  localStorage.setItem("song-sync-theme", theme);
};

applyTheme(preferredTheme);

themeToggleBtn.addEventListener("click", () => {
  applyTheme(document.body.classList.contains("dark") ? "light" : "dark");
});

document.addEventListener("keydown", (event) => {
  const now = Date.now();

  if (event.key === " " && event.target instanceof HTMLInputElement && event.target.type === "checkbox") {
    event.preventDefault();
    return;
  }

  if (!isRunning && event.key.toLowerCase() === "q") {
    isRunning = true;
    lastPressTime = null;
    startTime = null;
    delays = [];
    actions = [];
    holdStart = null;
    lastPulseEnd = null;
    totalElapsed = 0;
    log = ["Timer reset. Press spacebar to start again."];
    logDiv.textContent = log.join("\n");
    arduinoCodeDiv.textContent = "";
    copyBtn.style.display = "none";
    return;
  }

  if (!isRunning) return;

  if (event.key === " ") {
    event.preventDefault();

    if (startTime === null) {
      startTime = now;
      lastPressTime = now;
      log.push("Started tracking...");
    } else {
      const delay = cumulativeTimingCheckbox.checked ? now - startTime : now - lastPressTime;
      lastPressTime = now;
      delays.push(delay);
      actions.push({ type: "delay", value: delay });
      log.push(`Delay: ${delay} ms`);
    }
  }

  if (event.key && event.key.toLowerCase() === "q") {
    isRunning = false;
    totalElapsed = now - startTime;
    log.push(`\nStopped.\nTotal elapsed time: ${totalElapsed} ms`);
  }

  if (event.key && event.key.toLowerCase() === "p" && holdStart === null) {
    holdStart = now;
  }

  logDiv.textContent = log.join("\n");
});

document.addEventListener("keyup", (event) => {
  if (!isRunning) return;
  const now = Date.now();

  if (event.key && event.key.toLowerCase() === "p" && holdStart !== null) {
    const holdDuration = now - holdStart;
    if (lastPulseEnd !== null) {
      const delay = holdStart - lastPulseEnd;
      delays.push(delay);
      actions.push({ type: "delay", value: delay });
      log.push(`Delay: ${delay} ms`);
    }
    actions.push({ type: "pulse", value: holdDuration });
    log.push(`pulse(${holdDuration})`);
    lastPulseEnd = now;
    holdStart = null;
  }

  logDiv.textContent = log.join("\n");
});

exportBtn.addEventListener("click", () => {
  if (actions.length === 0) {
    arduinoCodeDiv.textContent = "No delays recorded yet.";
    return;
  }

  const extra = customTextInput.value.trim();
  const customDelayName = customDelayNameInput.value.trim() || "delay";
  const useOffset = enableOffsetCheckbox.checked;
  const offsetSign = offsetSignSelect.value;

  const timeList = actions
    .filter(action => action.type === "delay")
    .map(action => action.value)
    .join(",");

  const timeArrayLine = `const int times[] = {${timeList}};`;

  const code = actions.map(action => {
    if (action.type === "delay") {
      const delayExpr = useOffset ? `${action.value} ${offsetSign} offset` : action.value;
      return `${customDelayName}(${delayExpr});${extra ? " " + extra : ""}`;
    }

    if (action.type === "pulse") {
      return `pulse(${action.value});`;
    }

    return "";
  }).join("\n");

  arduinoCodeDiv.textContent = `${timeArrayLine}\n${code}`;
  copyBtn.style.display = "inline-block";
});

copyBtn.addEventListener("click", () => {
  const code = arduinoCodeDiv.textContent;
  if (!code) return;

  navigator.clipboard.writeText(code)
    .then(() => alert("Code copied to clipboard!"))
    .catch(err => alert("Failed to copy: " + err));
});
