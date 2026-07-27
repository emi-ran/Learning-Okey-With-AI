const params = new URLSearchParams(window.location.search);
const statusUrl = params.get("status")
  ?? "/artifacts/training/live-200/status.json";

const phaseLabels = {
  starting: "Trainer hazırlanıyor",
  training: "Self-play devam ediyor",
  checkpointing: "Model kaydediliyor",
  rendering: "Checkpoint filmi hazırlanıyor",
  finalizing: "Son kontroller yapılıyor",
  complete: "Eğitim tamamlandı",
  error: "Eğitim durdu",
};

const terminalLabels = {
  normal_finish: "Normal bitiş",
  same_turn_open_finish: "Aynı tur açıp bitiş",
  same_turn_open_okey_finish: "Aynı tur Okey bitişi",
  elden_finish: "Elden bitiş",
  okey_finish: "Okey ile bitiş",
  elden_okey_finish: "Elden Okey bitişi",
  pair_finish: "Çift bitişi",
  pair_okey_finish: "Çift + Okey bitişi",
  stock_exhausted: "Stok tükendi",
  all_players_opened_pairs: "Herkes çift açtı",
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

function formatInteger(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Math.round(Number(value)).toLocaleString("tr-TR");
}

function formatPercent(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `%${(Number(value) * 100).toLocaleString("tr-TR", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 1,
  })}`;
}

function formatDurationSeconds(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const seconds = Math.max(0, Math.round(Number(value)));
  if (seconds < 60) return `${seconds} sn`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) return `${hours} sa ${minutes} dk`;
  return `${minutes} dk ${seconds % 60} sn`;
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

function renderChart(history, historyTotal) {
  const visible = history.slice(-400);
  byId("lossLine").setAttribute(
    "points",
    chartPoints(visible.map((item) => item.loss), 830, 210, 45, 35),
  );
  byId("scoreLine").setAttribute(
    "points",
    chartPoints(visible.map((item) => item.mean_score), 830, 210, 45, 35),
  );
  byId("chartEmpty").style.display = visible.length ? "none" : "";
  byId("historyCoverage").textContent = historyTotal > history.length
    ? `Grafik için ${formatInteger(history.length)} örnek gösteriliyor; tam ${formatInteger(historyTotal)} episode JSONL raporunda saklı.`
    : `${formatInteger(history.length)} episode canlı grafikte.`;
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
    const label = document.createElement("b");
    label.textContent = formatInteger(episode);
    tick.append(label);
    container.append(tick);
  });
}

function renderEpisodeLog(history) {
  const container = byId("episodeLog");
  container.replaceChildren();
  const visible = history.slice(-9).reverse();
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
    const detail = document.createElement("span");
    const score = document.createElement("em");
    episode.textContent = `#${formatInteger(item.episode)}`;
    const penalties = item.penalty_events
      ? ` · ${item.penalty_events} ceza`
      : "";
    detail.textContent = `${item.actions} hamle${penalties} · ${
      terminalLabels[item.terminal_reason] ?? item.terminal_reason
    }`;
    score.textContent = `skor ${formatNumber(item.mean_score, 1)}`;
    row.append(episode, detail, score);
    container.append(row);
  });
}

function renderPlayers(players = []) {
  const body = byId("playerRows");
  body.replaceChildren();
  if (!players.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.textContent = "İlk episode bekleniyor.";
    row.append(cell);
    body.append(row);
    return;
  }
  players.forEach((player) => {
    const row = document.createElement("tr");
    const openings = player.opened_series + player.opened_pairs;
    const values = [
      `Oyuncu ${player.seat + 1}`,
      formatInteger(player.playable_discards),
      formatInteger(player.real_okey_discards),
      `${formatInteger(player.penalty_events)} / ${formatInteger(player.penalty_points)}`,
      formatInteger(openings),
      formatInteger(player.finishes),
      formatNumber(player.mean_score, 1),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index ? "td" : "th");
      if (!index) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    });
    if (player.penalty_events) row.dataset.penalized = "true";
    body.append(row);
  });
}

function renderTerminalReasons(reasons = {}, episodeCount = 0) {
  const container = byId("terminalReasons");
  container.replaceChildren();
  const entries = Object.entries(reasons).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.textContent = "Henüz tamamlanan episode yok.";
    container.append(empty);
    return;
  }
  const maximum = Math.max(...entries.map(([, count]) => count), 1);
  entries.forEach(([reason, count]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const value = document.createElement("b");
    const bar = document.createElement("i");
    label.textContent = terminalLabels[reason] ?? reason.replaceAll("_", " ");
    value.textContent = `${formatInteger(count)} · ${formatPercent(count / episodeCount)}`;
    bar.style.setProperty("--bar", `${(count / maximum) * 100}%`);
    row.append(label, value, bar);
    container.append(row);
  });
}

function renderTelemetry(telemetry = {}) {
  const performance = telemetry.performance ?? {};
  const totals = telemetry.totals ?? {};
  const rates = telemetry.rates ?? {};
  byId("liveSpeed").textContent = formatNumber(
    performance.rolling_episodes_per_second,
    2,
  );
  byId("actionSpeed").textContent = formatNumber(
    performance.actions_per_second,
    0,
  );
  byId("etaTime").textContent = formatDurationSeconds(performance.eta_seconds);
  byId("totalDiscards").textContent = `${formatInteger(totals.discard_actions ?? 0)} ATMA`;
  byId("playableDiscards").textContent = formatInteger(totals.playable_discards ?? 0);
  byId("playableRate").textContent = formatPercent(rates.playable_discard ?? 0);
  byId("okeyDiscards").textContent = formatInteger(totals.real_okey_discards ?? 0);
  byId("okeyRate").textContent = formatPercent(rates.real_okey_discard ?? 0);
  byId("penaltyEvents").textContent = formatInteger(totals.penalty_events ?? 0);
  byId("penaltyPoints").textContent = `${formatInteger(totals.immediate_penalty_points ?? 0)} puan`;
  byId("finishCount").textContent = formatInteger(totals.finishes ?? 0);
  byId("finishRate").textContent = formatPercent(rates.finish ?? 0);
  byId("openedSeries").textContent = formatInteger(totals.opened_series ?? 0);
  byId("openedPairs").textContent = formatInteger(totals.opened_pairs ?? 0);
  byId("penaltyMean").textContent = formatNumber(rates.penalty_per_episode ?? 0, 1);
  byId("totalActions").textContent = formatInteger(totals.actions ?? 0);
  renderPlayers(telemetry.players);
  renderTerminalReasons(telemetry.terminal_reasons, totals.episodes || 1);
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
    card.querySelector(".checkpoint-step").textContent = `CHECKPOINT ${formatInteger(episode)}`;
    card.querySelector(".checkpoint-name").textContent = `${formatInteger(episode)} episode`;
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
      const metrics = {
        score: formatNumber(evaluation.mean_score, 1),
        reward: formatNumber(evaluation.mean_relative_reward, 2),
        finish: formatPercent(evaluation.finish_rate),
        penalty: formatNumber(evaluation.mean_immediate_penalty, 1),
        penalized: formatPercent(evaluation.penalized_episode_rate),
        series: formatPercent(evaluation.opened_series / evaluation.episodes),
        playable: formatPercent(evaluation.playable_discard_rate),
        okey: formatPercent(evaluation.real_okey_discard_rate),
      };
      Object.entries(metrics).forEach(([key, value]) => {
        card.querySelector(`[data-metric="${key}"]`).textContent = value;
      });
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

  byId("phaseTitle").textContent = phaseLabels[status.state.phase]
    ?? status.state.phase;
  byId("phaseMessage").textContent = status.state.message;
  byId("episodeCurrent").textContent = formatInteger(current);
  byId("episodeTarget").textContent = formatInteger(target);
  byId("progressFill").style.width = `${progress * 100}%`;
  byId("progressPercent").textContent = formatPercent(progress);
  byId("runName").textContent = status.run.name;
  byId("trainSeed").textContent = status.run.seed;
  byId("replaySeed").textContent = status.run.replay_seed;
  byId("evaluationEpisodes").textContent = `${status.run.evaluation_episodes} EL`;
  byId("lastUpdated").textContent = new Date().toLocaleTimeString("tr-TR");
  byId("elapsedTime").textContent = formatElapsed(
    status.run.started_at,
    status.run.completed_at ? new Date(status.run.completed_at) : new Date(),
  );
  byId("checkpointSeedLabel").textContent =
    `SABİT REPLAY SEED ${status.run.replay_seed} · ${status.run.evaluation_episodes} DEĞERLENDİRME ELİ`;

  const next = status.run.checkpoint_episodes.find((episode) => episode > current);
  byId("nextCheckpoint").textContent = next == null
    ? "Tüm checkpointler tamamlandı"
    : `Sonraki checkpoint: ${formatInteger(next)}`;
  byId("lastLoss").textContent = formatNumber(last?.loss, 4);
  byId("lastScore").textContent = formatNumber(last?.mean_score, 1);
  byId("lastActions").textContent = last?.actions ?? "—";

  const historyLink = byId("historyDownload");
  if (status.run.history_path) {
    historyLink.href = status.run.history_path;
    historyLink.classList.remove("is-disabled");
  }

  setConnection(
    status.state.phase === "error" ? "error" : "live",
    status.state.phase === "complete" ? "EĞİTİM TAMAMLANDI" : "CANLI VERİ",
  );
  renderTicks(status);
  renderTelemetry(status.telemetry);
  renderChart(history, status.history_entries_total ?? history.length);
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
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
