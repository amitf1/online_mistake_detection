const state = {
  videos: [],
  currentVideo: null,
  moduleBGlobalPrediction: null,
};

const elements = {
  videoSelect: document.getElementById("videoSelect"),
  stepSelect: document.getElementById("stepSelect"),
  videoPlayer: document.getElementById("videoPlayer"),
  startRange: document.getElementById("startRange"),
  endRange: document.getElementById("endRange"),
  startLabel: document.getElementById("startLabel"),
  endLabel: document.getElementById("endLabel"),
  selectedRange: document.getElementById("selectedRange"),
  moduleBRange: document.getElementById("moduleBRange"),
  playClipButton: document.getElementById("playClipButton"),
  runAButton: document.getElementById("runAButton"),
  runBButton: document.getElementById("runBButton"),
  runCButton: document.getElementById("runCButton"),
  unloadModelButton: document.getElementById("unloadModelButton"),
  status: document.getElementById("status"),
  videoMetadata: document.getElementById("videoMetadata"),
  moduleAParsed: document.getElementById("moduleAParsed"),
  moduleARaw: document.getElementById("moduleARaw"),
  moduleBParsed: document.getElementById("moduleBParsed"),
  moduleBRaw: document.getElementById("moduleBRaw"),
  moduleCParsed: document.getElementById("moduleCParsed"),
  moduleCRaw: document.getElementById("moduleCRaw"),
};

function seconds(value) {
  return `${Number(value).toFixed(2)}s`;
}

function timingText(payload) {
  return `Prediction time: ${seconds(payload.prediction_seconds || 0)}\nRequest wall time: ${seconds(payload.request_wall_seconds || 0)}`;
}

function selectedClip() {
  return {
    video_id: elements.videoSelect.value,
    step_index: Number(elements.stepSelect.value),
    clip_start: Number(elements.startRange.value),
    clip_end: Number(elements.endRange.value),
  };
}

function setStatus(message) {
  elements.status.textContent = message;
}

function setBusy(isBusy) {
  for (const button of [
    elements.runAButton,
    elements.runBButton,
    elements.runCButton,
    elements.unloadModelButton,
  ]) {
    button.disabled = isBusy;
  }
}

function updateRangeLabels() {
  let start = Number(elements.startRange.value);
  let end = Number(elements.endRange.value);
  const step = Number(elements.startRange.step || 5);
  if (end <= start) {
    end = start + step;
    elements.endRange.value = String(end);
  }
  const duration = Math.max(Number(elements.videoPlayer.duration || state.currentVideo?.duration || end), end);
  elements.startLabel.textContent = seconds(start);
  elements.endLabel.textContent = seconds(end);
  const left = duration > 0 ? (start / duration) * 100 : 0;
  const width = duration > 0 ? ((end - start) / duration) * 100 : 0;
  elements.selectedRange.style.left = `${left}%`;
  elements.selectedRange.style.width = `${width}%`;
}

function updateModuleBOverlay() {
  if (!state.moduleBGlobalPrediction || !state.currentVideo) {
    elements.moduleBRange.classList.add("hidden");
    return;
  }
  const [start, end] = state.moduleBGlobalPrediction;
  const duration = Number(elements.videoPlayer.duration || state.currentVideo.duration || end);
  const left = duration > 0 ? (start / duration) * 100 : 0;
  const width = duration > 0 ? ((end - start) / duration) * 100 : 0;
  elements.moduleBRange.style.left = `${left}%`;
  elements.moduleBRange.style.width = `${width}%`;
  elements.moduleBRange.classList.remove("hidden");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json();
}

async function loadVideos() {
  const payload = await fetchJson("/api/videos");
  state.videos = payload.videos;
  elements.videoSelect.innerHTML = "";
  for (const video of state.videos) {
    const option = document.createElement("option");
    option.value = video.video_id;
    option.textContent = `${video.task_id} / ${video.video_id}`;
    elements.videoSelect.appendChild(option);
  }
  if (state.videos.length) {
    await loadVideo(state.videos[0].video_id);
  } else {
    setStatus("No videos found at the configured video root.");
  }
}

async function loadVideo(videoId) {
  const payload = await fetchJson(`/api/videos/${encodeURIComponent(videoId)}`);
  state.currentVideo = payload.video;
  state.moduleBGlobalPrediction = null;
  elements.moduleBRange.classList.add("hidden");
  elements.videoPlayer.src = state.currentVideo.video_url;
  elements.videoMetadata.textContent = JSON.stringify(state.currentVideo, null, 2);
  elements.stepSelect.innerHTML = "";
  for (const instruction of state.currentVideo.instructions) {
    const option = document.createElement("option");
    option.value = instruction.step_index;
    option.textContent = `${instruction.step_index}: ${instruction.text}`;
    elements.stepSelect.appendChild(option);
  }
  configureRanges(state.currentVideo.duration, state.currentVideo.step_seconds || 5);
  resetResults();
  setStatus("Ready.");
}

function configureRanges(duration, stepSeconds) {
  const max = Math.max(stepSeconds, Math.floor(Number(duration || stepSeconds) / stepSeconds) * stepSeconds);
  for (const range of [elements.startRange, elements.endRange]) {
    range.step = String(stepSeconds);
    range.min = "0";
    range.max = String(max);
  }
  elements.startRange.value = "0";
  elements.endRange.value = String(Math.min(stepSeconds, max));
  updateRangeLabels();
}

function resetResults() {
  state.moduleBGlobalPrediction = null;
  elements.moduleAParsed.textContent = "Not run";
  elements.moduleARaw.textContent = "";
  elements.moduleBParsed.textContent = "Not run";
  elements.moduleBRaw.textContent = "";
  elements.moduleCParsed.textContent = "Not run";
  elements.moduleCRaw.textContent = "";
  updateModuleBOverlay();
}

function playSelectedClip() {
  const clip = selectedClip();
  elements.videoPlayer.currentTime = clip.clip_start;
  elements.videoPlayer.play();
}

function stopAtClipEnd() {
  const clip = selectedClip();
  if (elements.videoPlayer.currentTime >= clip.clip_end) {
    elements.videoPlayer.pause();
    elements.videoPlayer.currentTime = clip.clip_end;
  }
}

async function runModuleA() {
  await runModule("/api/module-a", selectedClip(), elements.moduleAParsed, elements.moduleARaw, (payload) => {
    return `Prediction: ${payload.prediction}\n${timingText(payload)}`;
  });
}

async function runModuleB() {
  await runModule("/api/module-b", selectedClip(), elements.moduleBParsed, elements.moduleBRaw, (payload) => {
    state.moduleBGlobalPrediction = payload.global_prediction;
    updateModuleBOverlay();
    if (!payload.prediction) {
      return `Prediction: not completed / invalid\n${timingText(payload)}`;
    }
    return `Local window: ${payload.prediction.map(seconds).join(" - ")}\nGlobal window: ${payload.global_prediction.map(seconds).join(" - ")}\n${timingText(payload)}`;
  });
}

async function runModuleC() {
  const clip = selectedClip();
  const body = {
    ...clip,
    predicted_start: state.moduleBGlobalPrediction ? state.moduleBGlobalPrediction[0] : null,
    predicted_end: state.moduleBGlobalPrediction ? state.moduleBGlobalPrediction[1] : null,
  };
  await runModule("/api/module-c", body, elements.moduleCParsed, elements.moduleCRaw, (payload) => {
    if (!payload.prediction) {
      return `Prediction: invalid JSON\n${timingText(payload)}`;
    }
    return `Mistake: ${payload.prediction.mistake}\nReasoning: ${payload.prediction.reasoning}\n${timingText(payload)}`;
  });
}

async function runModule(url, body, parsedElement, rawElement, renderParsed) {
  setBusy(true);
  setStatus(`Running ${url.split("/").pop()}... first call may load the checkpoint.`);
  try {
    const payload = await fetchJson(url, {
      method: "POST",
      body: JSON.stringify(body),
    });
    parsedElement.textContent = renderParsed(payload);
    rawElement.textContent = JSON.stringify(payload, null, 2);
    setStatus("Ready.");
  } catch (error) {
    parsedElement.textContent = "Error";
    rawElement.textContent = String(error);
    setStatus("Error. See result panel.");
  } finally {
    setBusy(false);
  }
}

async function unloadModel() {
  setBusy(true);
  try {
    await fetchJson("/api/unload-model", { method: "POST", body: "{}" });
    setStatus("GPU model unloaded.");
  } finally {
    setBusy(false);
  }
}

elements.videoSelect.addEventListener("change", () => loadVideo(elements.videoSelect.value));
elements.stepSelect.addEventListener("change", resetResults);
elements.startRange.addEventListener("input", () => {
  updateRangeLabels();
  resetResults();
});
elements.endRange.addEventListener("input", () => {
  updateRangeLabels();
  resetResults();
});
elements.videoPlayer.addEventListener("timeupdate", stopAtClipEnd);
elements.videoPlayer.addEventListener("loadedmetadata", () => {
  updateRangeLabels();
  updateModuleBOverlay();
});
elements.playClipButton.addEventListener("click", playSelectedClip);
elements.runAButton.addEventListener("click", runModuleA);
elements.runBButton.addEventListener("click", runModuleB);
elements.runCButton.addEventListener("click", runModuleC);
elements.unloadModelButton.addEventListener("click", unloadModel);

loadVideos().catch((error) => {
  setStatus("Failed to load videos.");
  elements.videoMetadata.textContent = String(error);
});
