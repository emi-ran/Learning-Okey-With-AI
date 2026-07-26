const params = new URLSearchParams(window.location.search);
const statusUrl = params.get("status") ?? "/training_runs/live-200/status.json";
const phaseLabels = {
  starting: "Trainer hazırlanıyor",
  training: "Self-play devam ediyor",
  checkpointing: "Model mühürleniyor",
  rendering: "Checkpoint filmi hazırlanıyor",
  finalizing: "Son kontroller yapılıyor",
  complete: "200 episode tamamlandı",
  error: "Eğitim durdu",
};

let latestStatus = null;
let lastFetchAt = null;
let checkpointSignature = "";

const byId = (id) => document.getElementById(id);

function formatNumber(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString("tr-TR", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatPercent(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatElapsed(start, end = new Date()) {
  if (!start) return "00:00";
  const seconds = Math.max(
    0,
    Math.floor((new Date(end).getTime() - new Date(start).getTime()) / 1000),
  );
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function setConnection(state, label) {
  const badge = byId("connectionBadge");
  badge.dataset.state = state;
  byId("connectionText").textContent = label;
}

function chartPoints(values, width, height, left, top) {
  if (!values.length) return "";
  const finite = values.map(Number).filter(Number.isFinite);
  if (!finite.length) return "";
  const minimum = Math.min(...finite);
  const maximum = Math.max(...finite);
  const range = maximum - minimum || 1;
  return values
    .map((value, index) => {
      const x = left + (index / Math.max(1, values.length - 1)) * width;
      const y = top + height - ((Number(value) - minimum) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function renderChart(history) {
  const visible = history.slice(-200);
  byId("lossLine").setAttribute(
    "points",
    chartPoints(
      visible.map((item) => item.loss),
      830,
      210,
      45,
      35,
    ),
  );
  byId("scoreLine").setAttribute(
    "points",
    chartPoints(
      visible.map((item) => item.mean_score),
      830,
      210,
      45,
      35,
    ),
  );
  byId("chartEmpty").style.display = visible.length ? "none" : "";
}

function renderTicks(status) {
  const container = byId("checkpointTicks");
  const target = status.run.target_episodes;
  const ready = new Set(
    status.checkpoints
      .filter((item) => item.status === "ready")
      .map((item) => item.episode),
  );
  container.replaceChildren();
  status.run.checkpoint_episodes.forEach((episode) => {
    const tick = document.createElement("span");
    tick.style.left = `${target ? (episode / target) * 100 : 0}%`;
    tick.classList.toggle("ready", ready.has(episode));
    tick.title = `${episode} episode`;
    container.append(tick);
  });
}

function renderEpisodeLog(history) {
  const container = byId("episodeLog");
  container.replaceChildren();
  const visible = history.slice(-8).reverse();
  if (!visible.length) {
    const row = document.createElement("li");
    row.className = "empty-row";
    row.textContent = "Henüz tamamlanan episode yok.";
    container.append(row);
    return;
  }
  visible.forEach((item) => {
    const row = document.createElement("li");
    const episode = document.createElement("b");
    const terminal = document.createElement("span");
    const score = document.createElement("em");
    episode.textContent = `#${item.episode}`;
    terminal.textContent = `${item.actions} hamle · ${String(item.terminal_reason).replaceAll("_", " ")}`;
    score.textContent = `skor ${formatNumber(item.mean_score, 1)}`;
    row.append(episode, terminal, score);
    container.append(row);
  });
}

function checkpointKey(checkpoints) {
  return checkpoints
    .map((item) => `${item.episode}:${item.status}:${item.video_path ?? ""}`)
    .join("|");
}

function replayViewerUrl(path) {
  return `/viewer/?replay=${encodeURIComponent(path)}`;
}

function renderCheckpoints(status) {
  const signature = checkpointKey(status.checkpoints);
  if (signature === checkpointSignature) return;
  checkpointSignature = signature;
  const container = byId("checkpointGrid");
  const template = byId("checkpointTemplate");
  const checkpointByEpisode = new Map(
    status.checkpoints.map((item) => [item.episode, item]),
  );
  container.replaceChildren();

  status.run.checkpoint_episodes.forEach((episode) => {
    const checkpoint = checkpointByEpisode.get(episode);
    const card = template.content.firstElementChild.cloneNode(true);
    const cardStatus = checkpoint?.status ?? "pending";
    card.dataset.status = cardStatus;
    card.querySelector(".checkpoint-step").textContent = `CHECKPOINT ${episode}`;
    card.querySelector(".checkpoint-name").textContent = `${episode} episode`;
    card.querySelector(".checkpoint-status").textContent = {
      pending: "BEKLİYOR",
      working: "HAZIRLANIYOR",
      ready: "HAZIR",
    }[cardStatus] ?? cardStatus.toUpperCase();

    if (checkpoint?.video_path) {
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.src = checkpoint.video_path;
      if (checkpoint.poster_path) video.poster = checkpoint.poster_path;
      card.querySelector(".video-shell").replaceChildren(video);
    }

    const evaluation = checkpoint?.evaluation;
    if (evaluation) {
      card.querySelector('[data-metric="score"]').textContent = formatNumber(
        evaluation.mean_score,
        1,
      );
      card.querySelector('[data-metric="reward"]').textContent = formatNumber(
        evaluation.mean_relative_reward,
        2,
      );
      card.querySelector('[data-metric="playable"]').textContent = formatPercent(
        evaluation.playable_discard_rate,
      );
      card.querySelector('[data-metric="okey"]').textContent = formatPercent(
        evaluation.real_okey_discard_rate,
      );
    }

    const replayLink = card.querySelector(".replay-link");
    if (checkpoint?.replay_path) {
      replayLink.href = replayViewerUrl(checkpoint.replay_path);
      replayLink.classList.remove("is-disabled");
    }
    const downloadLink = card.querySelector(".download-link");
    if (checkpoint?.video_path) {
      downloadLink.href = checkpoint.video_path;
      downloadLink.classList.remove("is-disabled");
    }
    container.append(card);
  });
}

function render(status) {
  latestStatus = status;
  const history = status.history ?? [];
  const current = status.state.current_episode ?? 0;
  const target = status.run.target_episodes ?? status.state.target_episodes ?? 0;
  const progress = target ? Math.min(1, current / target) : 0;
  const last = history.at(-1);

  byId("phaseTitle").textContent = status.state.phase === "complete"
    ? `${target} episode tamamlandı`
    : (phaseLabels[status.state.phase] ?? status.state.phase);
  byId("phaseMessage").textContent = status.state.message;
  byId("episodeCurrent").textContent = current;
  byId("episodeTarget").textContent = target;
  byId("progressFill").style.width = `${progress * 100}%`;
  byId("progressPercent").textContent = `${(progress * 100).toFixed(1)}%`;
  byId("runName").textContent = status.run.name;
  byId("trainSeed").textContent = status.run.seed;
  byId("replaySeed").textContent = status.run.replay_seed;
  byId("lastUpdated").textContent = new Date().toLocaleTimeString("tr-TR");
  byId("elapsedTime").textContent = formatElapsed(
    status.run.started_at,
    status.run.completed_at ? new Date(status.run.completed_at) : new Date(),
  );

  const next = status.run.checkpoint_episodes.find((episode) => episode > current);
  byId("nextCheckpoint").textContent = next == null
    ? "Tüm checkpointler tamamlandı"
    : `Sonraki checkpoint: ${next}`;
  byId("lastLoss").textContent = formatNumber(last?.loss, 4);
  byId("lastScore").textContent = formatNumber(last?.mean_score, 1);
  byId("lastGradient").textContent = formatNumber(last?.gradient_norm, 3);
  byId("lastActions").textContent = last?.actions ?? "—";

  setConnection(
    status.state.phase === "error" ? "error" : "live",
    status.state.phase === "complete" ? "EĞİTİM TAMAMLANDI" : "CANLI VERİ",
  );
  renderTicks(status);
  renderChart(history);
  renderEpisodeLog(history);
  renderCheckpoints(status);
}

async function poll() {
  try {
    const separator = statusUrl.includes("?") ? "&" : "?";
    const response = await fetch(
      `${statusUrl}${separator}t=${Date.now()}`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    lastFetchAt = new Date();
    render(payload);
  } catch (error) {
    if (!latestStatus) {
      setConnection("waiting", "TRAINER BEKLENİYOR");
      byId("phaseMessage").textContent =
        `Durum dosyası henüz yok: ${statusUrl}`;
    } else {
      setConnection("error", "BAĞLANTI KESİLDİ");
    }
  }
}

setInterval(() => {
  if (latestStatus?.run?.started_at) {
    const end = latestStatus.run.completed_at
      ? new Date(latestStatus.run.completed_at)
      : new Date();
    byId("elapsedTime").textContent = formatElapsed(
      latestStatus.run.started_at,
      end,
    );
  }
  if (lastFetchAt && Date.now() - lastFetchAt.getTime() > 5000) {
    setConnection("error", "VERİ GECİKTİ");
  }
}, 1000);

poll();
setInterval(poll, 900);
