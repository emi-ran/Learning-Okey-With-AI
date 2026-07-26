const ACTION_LABELS = {
  deal: "Taşlar dağıtıldı",
  draw_from_stock: "Ortadan taş çekti",
  draw_stock: "Ortadan taş çekti",
  take_previous_discard: "Yandaki taşı aldı",
  take_discard: "Yandaki taşı aldı",
  open_melds: "Serilerini açtı",
  open_series: "101 açtı",
  lay_melds: "Yeni per indirdi",
  open_pairs: "Çift açtı",
  add_to_meld: "Masadaki pere taş işledi",
  add_pair: "Çift alanına çift ekledi",
  replace_joker: "Okeyi geri aldı",
  end_table_actions: "Masa hamlelerini bitirdi",
  discard: "Taş attı",
  penalty: "Ceza aldı",
  finish: "Eli bitirdi",
  round_end: "El sona erdi",
};

const PHASE_LABELS = {
  draw_decision: "ÇEKME KARARI",
  table_actions: "MASA HAMLELERİ",
  discard: "TAŞ ATMA",
  terminal: "EL SONU",
};

const MODE_LABELS = {
  none: "AÇMADI",
  series: "SERİ AÇTI",
  pairs: "ÇİFT AÇTI",
};

const COLOR_ORDER = { red: 0, yellow: 1, blue: 2, black: 3 };

const demoReplay = {
  schema_version: 1,
  metadata: {
    title: "Checkpoint 0 — Rastgele politika",
    checkpoint_label: "BAŞLANGIÇ",
    training_step: 0,
    seed: 42,
    spectator: true,
  },
  metrics: {
    okey_discard_rate: 0.18,
    playable_penalty_rate: 0.31,
    finish_rate: 0.045,
    average_score: 171.8,
    history: [
      { step: 0, score: -0.05 },
      { step: 10000, score: 0.02 },
      { step: 50000, score: 0.11 },
      { step: 100000, score: 0.19 },
      { step: 250000, score: 0.33 },
    ],
  },
  frames: [
    {
      index: 0,
      round_id: 1,
      turn_number: 1,
      phase: "table_actions",
      current_player: 0,
      stock_count: 21,
      indicator: { color: "red", number: 13 },
      discard_top: null,
      action: { type: "deal", player_id: null, narration: "Sabit değerlendirme eli dağıtıldı" },
      players: [
        { seat: 0, name: "AI Güney", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("red") },
        { seat: 1, name: "AI Batı", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("yellow") },
        { seat: 2, name: "AI Kuzey", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("blue") },
        { seat: 3, name: "AI Doğu", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("black") },
      ],
      table: { melds: [], pairs: [] },
      policy: { value: -0.08, candidates: [] },
      events: [{ type: "deal", narration: "106 fiziksel taş deterministik seed ile karıldı." }],
    },
    {
      index: 1,
      round_id: 1,
      turn_number: 1,
      phase: "discard",
      current_player: 0,
      stock_count: 21,
      indicator: { color: "red", number: 13 },
      discard_top: null,
      action: { type: "end_table_actions", player_id: 0, narration: "AI Güney masa hamlelerini bitirdi" },
      players: [
        { seat: 0, name: "AI Güney", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("red") },
        { seat: 1, name: "AI Batı", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("yellow") },
        { seat: 2, name: "AI Kuzey", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("blue") },
        { seat: 3, name: "AI Doğu", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("black") },
      ],
      table: { melds: [], pairs: [] },
      policy: {
        value: -0.11,
        selected_index: 0,
        candidates: [
          { label: "Masa hamlelerini bitir", probability: 0.52 },
          { label: "Kırmızı seri aç", probability: 0.31 },
          { label: "Çift aç", probability: 0.17 },
        ],
      },
      events: [
        { type: "deal", narration: "El başladı." },
        { type: "end_table_actions", narration: "AI Güney açmadan taş atma aşamasına geçti." },
      ],
    },
    {
      index: 2,
      round_id: 1,
      turn_number: 1,
      phase: "draw_decision",
      current_player: 1,
      stock_count: 21,
      indicator: { color: "red", number: 13 },
      discard_top: { color: "red", number: 1, is_real_okey: true },
      action: {
        type: "discard",
        player_id: 0,
        narration: "AI Güney gerçek Okeyi attı",
        penalty: 101,
      },
      players: [
        { seat: 0, name: "AI Güney", opened_mode: "none", score: 0, penalty: 101, hand: demoHand("red").slice(1) },
        { seat: 1, name: "AI Batı", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("yellow") },
        { seat: 2, name: "AI Kuzey", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("blue") },
        { seat: 3, name: "AI Doğu", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("black") },
      ],
      table: { melds: [], pairs: [] },
      policy: {
        value: -0.82,
        selected_index: 0,
        candidates: [
          { label: "Gerçek Okeyi at", probability: 0.38, selected: true },
          { label: "Kırmızı 11'i at", probability: 0.35 },
          { label: "Mavi 4'ü at", probability: 0.27 },
        ],
      },
      events: [
        { type: "discard", narration: "AI Güney gerçek Okeyi attı." },
        { type: "penalty", narration: "Normal Okey atma cezası: +101." },
      ],
    },
    {
      index: 3,
      round_id: 1,
      turn_number: 2,
      phase: "table_actions",
      current_player: 1,
      stock_count: 20,
      indicator: { color: "red", number: 13 },
      discard_top: { color: "red", number: 1, is_real_okey: true },
      action: { type: "draw_stock", player_id: 1, narration: "AI Batı ortadan taş çekti" },
      players: [
        { seat: 0, name: "AI Güney", opened_mode: "none", score: 0, penalty: 101, hand: demoHand("red").slice(1) },
        { seat: 1, name: "AI Batı", opened_mode: "none", score: 0, penalty: 0, hand: [...demoHand("yellow"), { color: "blue", number: 9 }] },
        { seat: 2, name: "AI Kuzey", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("blue") },
        { seat: 3, name: "AI Doğu", opened_mode: "none", score: 0, penalty: 0, hand: demoHand("black") },
      ],
      table: {
        melds: [
          {
            kind: "run",
            tiles: [
              { color: "yellow", number: 5 },
              { color: "yellow", number: 6 },
              { color: "yellow", number: 7 },
            ],
          },
        ],
        pairs: [],
      },
      policy: {
        value: 0.06,
        selected_index: 1,
        candidates: [
          { label: "Yandaki Okeyi al", probability: 0.42 },
          { label: "Ortadan çek", probability: 0.58, selected: true },
        ],
      },
      events: [
        { type: "penalty", narration: "AI Güney +101 ceza aldı." },
        { type: "draw_stock", narration: "AI Batı ortadan taş çekti." },
      ],
    },
  ],
};

function demoHand(primary) {
  const colors = ["red", "yellow", "blue", "black"];
  const offset = COLOR_ORDER[primary] ?? 0;
  return Array.from({ length: 12 }, (_, index) => ({
    color: colors[(index + offset) % colors.length],
    number: (index * 3 + offset) % 13 + 1,
    is_real_okey: index === 0 && primary === "red",
  })).sort(tileSort);
}

let replay = demoReplay;
let frameIndex = 0;
let playing = false;
let timer = null;
let spectator = true;

const byId = (id) => document.getElementById(id);

function tileSort(a, b) {
  return (COLOR_ORDER[a.color] ?? 9) - (COLOR_ORDER[b.color] ?? 9)
    || (a.number ?? 99) - (b.number ?? 99);
}

function createTile(tile) {
  if (!tile) return document.createTextNode("—");
  const source = tile.tile ?? tile;
  const display = source.display ?? source.physical ?? source.value ?? source;
  const node = byId("tileTemplate").content.firstElementChild.cloneNode(true);
  const number = node.querySelector(".tile-number");
  const kind = source.kind ?? source.tile_kind;
  const color = display.color ?? source.color ?? "black";
  const value = display.number ?? source.number;
  node.classList.add(color);

  if (kind === "fake_okey" || source.is_fake_okey) {
    node.classList.add("fake");
    number.textContent = "SAHTE";
  } else {
    number.textContent = value ?? "★";
  }
  if (source.is_real_okey) node.classList.add("real-okey");
  if (tile.represented_value) {
    node.title = `Temsil: ${tile.represented_value.color} ${tile.represented_value.number}`;
  }
  return node;
}

function normalizeReplay(payload) {
  const frames = payload.frames ?? payload.timeline ?? payload.steps;
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error("Replay dosyasında en az bir frame bulunmalı.");
  }
  const checkpoint = payload.checkpoint ?? {};
  const episode = payload.episode ?? {};
  return {
    schema_version: payload.schema_version ?? payload.version ?? 1,
    metadata: {
      ...(payload.metadata ?? payload.meta ?? {}),
      checkpoint_label: checkpoint.label ?? checkpoint.id,
      checkpoint: checkpoint.id,
      training_step: checkpoint.training_step,
      seed: episode.seed,
      title: payload.summary?.title
        ?? `${checkpoint.label ?? checkpoint.id ?? "Replay"} · seed ${episode.seed ?? "—"}`,
      policy_name: payload.policy?.name,
    },
    metrics: payload.metrics ?? payload.summary?.metrics ?? {},
    frames,
  };
}

function normalizeFrame(raw, index) {
  const state = raw.view ?? raw.state ?? raw;
  const actionPayload = raw.action ?? raw.decision?.action ?? raw.events?.[0] ?? {};
  const action = {
    ...actionPayload,
    player_id: raw.actor_seat ?? actionPayload.player_id,
    narration: raw.narration ?? actionPayload.narration,
  };
  return {
    index: raw.frame_index ?? raw.index ?? index,
    round_id: state.round_id ?? 1,
    turn_number: state.turn_number ?? 0,
    phase: state.phase ?? "table_actions",
    current_player: state.current_player ?? action.player_id ?? 0,
    stock_count: state.stock_count ?? state.stock?.length ?? 0,
    indicator: state.indicator,
    discard_top: state.discard_top ?? state.discard_pile?.at?.(-1) ?? null,
    players: state.players ?? [],
    table: state.table ?? { melds: [], pairs: [] },
    action,
    policy: raw.policy_step ?? raw.policy ?? raw.decision ?? {},
    events: raw.events ?? [],
  };
}

function render() {
  const raw = replay.frames[frameIndex];
  const frame = normalizeFrame(raw, frameIndex);
  const meta = replay.metadata;

  byId("checkpointLabel").textContent = meta.checkpoint_label ?? meta.checkpoint ?? "CHECKPOINT";
  byId("trainingStep").textContent = formatNumber(meta.training_step ?? 0);
  byId("seedLabel").textContent = meta.seed ?? frame.seed ?? "—";
  byId("episodeTitle").textContent = meta.title ?? "Model gelişim replay’i";
  byId("roundLabel").textContent = String(frame.round_id).padStart(2, "0");
  byId("turnLabel").textContent = String(frame.turn_number).padStart(2, "0");
  byId("stockCount").textContent = frame.stock_count;
  byId("phaseLabel").textContent = PHASE_LABELS[frame.phase] ?? String(frame.phase).toUpperCase();
  byId("frameCounter").textContent = `${frameIndex + 1} / ${replay.frames.length}`;
  byId("actionIndex").textContent = String(frame.index).padStart(3, "0");
  byId("timeline").max = Math.max(0, replay.frames.length - 1);
  byId("timeline").value = frameIndex;

  renderSingleTile(byId("indicatorTile"), frame.indicator);
  renderSingleTile(byId("discardTile"), frame.discard_top);
  renderPlayers(frame);
  renderTable(frame.table);
  renderAction(frame);
  renderCandidates(frame.policy);
  renderEvents(frame);
  renderMetrics(replay.metrics);

  const table = byId("okeyTable");
  table.classList.remove("frame-enter");
  requestAnimationFrame(() => table.classList.add("frame-enter"));
}

function renderSingleTile(container, tile) {
  container.replaceChildren(tile ? createTile(tile) : document.createTextNode("—"));
}

function renderPlayers(frame) {
  const players = [...frame.players].sort((a, b) => (a.seat ?? 0) - (b.seat ?? 0));
  document.querySelectorAll(".seat").forEach((seatNode) => {
    const seat = Number(seatNode.dataset.seat);
    const player = players.find((item) => (item.seat ?? item.player_id) === seat)
      ?? players[seat]
      ?? { seat, hand: [] };
    const hand = player.hand ?? player.tiles ?? [];
    const reveal = spectator || seat === frame.current_player;
    seatNode.classList.toggle("active", seat === frame.current_player);
    seatNode.innerHTML = `
      <div class="seat-heading">
        <div class="seat-identity">
          <span class="seat-number">${seat + 1}</span>
          <div>
            <h3>${escapeHtml(player.name ?? `AI Oyuncu ${seat + 1}`)}</h3>
            <span class="seat-mode">${MODE_LABELS[player.opened_mode] ?? "AÇMADI"}</span>
          </div>
        </div>
        <div class="seat-stats">
          <span>SKOR <strong>${player.score ?? 0}</strong></span>
          <span>CEZA <strong>${player.penalty ?? player.immediate_penalty ?? 0}</strong></span>
          <span>TAŞ <strong>${hand.length || player.hand_count || 0}</strong></span>
        </div>
      </div>
      <div class="rack${reveal ? "" : " concealed"}"></div>
    `;
    const rack = seatNode.querySelector(".rack");
    if (reveal) {
      [...hand].sort(tileSort).forEach((tile) => rack.append(createTile(tile)));
    } else {
      Array.from({ length: hand.length || player.hand_count || 0 }, () => {
        const back = document.createElement("i");
        back.className = "tile-back";
        rack.append(back);
      });
    }
  });
}

function renderTable(table) {
  const melds = table?.melds ?? [];
  const pairs = table?.pairs ?? table?.pair_area ?? [];
  byId("meldCount").textContent = melds.length;
  byId("pairCount").textContent = pairs.length;
  const meldContainer = byId("tableMelds");
  const pairContainer = byId("pairArea");
  meldContainer.replaceChildren();
  pairContainer.replaceChildren();
  meldContainer.classList.toggle("empty-state", melds.length === 0);
  pairContainer.classList.toggle("empty-state", pairs.length === 0);

  if (melds.length === 0) meldContainer.textContent = "Henüz per açılmadı";
  melds.forEach((meld) => {
    const node = document.createElement("div");
    node.className = "meld";
    (meld.tiles ?? meld.meld?.tiles ?? []).forEach((tile) => node.append(createTile(tile.tile ?? tile.physical_tile ?? tile)));
    meldContainer.append(node);
  });

  if (pairs.length === 0) pairContainer.textContent = "—";
  pairs.forEach((pair) => {
    const node = document.createElement("div");
    node.className = "pair";
    (pair.tiles ?? pair).forEach((tile) => node.append(createTile(tile.tile ?? tile.physical_tile ?? tile)));
    pairContainer.append(node);
  });
}

function renderAction(frame) {
  const action = frame.action ?? {};
  const type = action.type ?? action.action_type ?? frame.events.at?.(-1)?.type ?? "deal";
  const actor = action.player_id ?? action.seat;
  byId("actionActor").textContent = actor == null ? "SİSTEM" : `AI OYUNCU ${actor + 1}`;
  byId("actionNarration").textContent = action.narration
    ?? action.label
    ?? ACTION_LABELS[type]
    ?? String(type).replaceAll("_", " ");
  const penalty = action.penalty
    ?? frame.events.find((event) => event.type === "penalty")?.details?.amount
    ?? 0;
  const chip = byId("actionPenalty");
  chip.classList.toggle("hidden", !penalty);
  chip.textContent = `+${penalty} CEZA`;
}

function renderCandidates(policy) {
  const candidates = policy?.candidates ?? policy?.top_candidates ?? [];
  const selectedProbability = policy?.selected_probability;
  const selected = policy?.selected_index ?? candidates.findIndex((item) => (
    item.selected
    || (
      selectedProbability != null
      && Math.abs((item.probability ?? item.prob ?? 0) - selectedProbability) < 1e-12
    )
  ));
  byId("stateValue").textContent = formatSigned(policy?.value ?? policy?.state_value ?? 0);
  const container = byId("candidateList");
  container.replaceChildren();

  if (!candidates.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Bu frame için politika dağılımı kaydedilmedi.";
    container.append(empty);
    return;
  }

  const visible = [...candidates]
    .map((item, index) => ({ ...item, originalIndex: index }))
    .sort((a, b) => (b.probability ?? b.prob ?? 0) - (a.probability ?? a.prob ?? 0))
    .slice(0, 6);
  const maxProbability = Math.max(...visible.map((item) => item.probability ?? item.prob ?? 0), 0.001);
  visible.forEach((candidate) => {
    const probability = candidate.probability ?? candidate.prob ?? 0;
    const row = document.createElement("div");
    row.className = `candidate-row${candidate.originalIndex === selected || candidate.selected ? " selected" : ""}`;
    row.innerHTML = `
      <div class="candidate-label">
        <span>${escapeHtml(candidateLabel(candidate))}</span>
        <span>${(probability * 100).toFixed(1)}%</span>
      </div>
      <div class="candidate-bar"><i style="width:${Math.max(2, probability / maxProbability * 100)}%"></i></div>
    `;
    container.append(row);
  });
}

function candidateLabel(candidate) {
  if (candidate.label || candidate.narration) {
    return candidate.label ?? candidate.narration;
  }
  const action = candidate.action ?? candidate;
  const base = ACTION_LABELS[action.type] ?? String(action.type ?? "aday").replaceAll("_", " ");
  if (action.tile_id != null) return `${base} · taş #${action.tile_id}`;
  if (action.meld_id != null) return `${base} · per #${action.meld_id}`;
  return base;
}

function renderEvents(frame) {
  const currentEvents = frame.events.length
    ? frame.events
    : [{ type: frame.action?.type ?? "deal", narration: frame.action?.narration }];
  const history = replay.frames
    .slice(Math.max(0, frameIndex - 4), frameIndex)
    .flatMap((item) => item.events ?? [item.action].filter(Boolean));
  const events = [...history, ...currentEvents].slice(-6);
  const container = byId("eventLog");
  container.replaceChildren();
  events.forEach((event, index) => {
    const li = document.createElement("li");
    if (index >= events.length - currentEvents.length) li.className = "current-event";
    const label = event.narration ?? event.label ?? ACTION_LABELS[event.type] ?? event.type ?? "Olay";
    li.innerHTML = `<b>${String(Math.max(0, frame.index - events.length + index + 1)).padStart(2, "0")}</b><span>${escapeHtml(label)}</span>`;
    container.append(li);
  });
}

function renderMetrics(metrics) {
  byId("okeyDiscardMetric").textContent = formatPercent(metrics.okey_discard_rate);
  byId("playablePenaltyMetric").textContent = formatPercent(metrics.playable_penalty_rate);
  byId("finishRateMetric").textContent = formatPercent(metrics.finish_rate);
  byId("averageScoreMetric").textContent = metrics.average_score == null ? "—" : Number(metrics.average_score).toFixed(1);
  drawMetricChart(metrics.history ?? []);
}

function drawMetricChart(history) {
  const svg = byId("metricChart");
  const width = 480;
  const height = 150;
  const pad = 14;
  const values = history.length
    ? history.map((item) => Number(item.score ?? item.value ?? 0))
    : [0, 0.04, 0.02, 0.11, 0.16];
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0.01);
  const range = max - min || 1;
  const points = values.map((value, index) => {
    const x = pad + index / Math.max(1, values.length - 1) * (width - pad * 2);
    const y = height - pad - (value - min) / range * (height - pad * 2);
    return [x, y];
  });
  const line = points.map((point) => point.join(",")).join(" ");
  const area = `${pad},${height - pad} ${line} ${width - pad},${height - pad}`;
  svg.innerHTML = `
    <defs>
      <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#c8a45a" stop-opacity=".32"/>
        <stop offset="100%" stop-color="#c8a45a" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <line class="grid-line" x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}"/>
    <polygon class="chart-area" points="${area}"/>
    <polyline class="chart-line" points="${line}"/>
    ${points.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="4"/>`).join("")}
  `;
}

function setFrame(next) {
  frameIndex = Math.max(0, Math.min(replay.frames.length - 1, Number(next)));
  render();
}

function togglePlay() {
  playing = !playing;
  byId("playIcon").textContent = playing ? "Ⅱ" : "▶";
  byId("playPause").setAttribute("aria-label", playing ? "Duraklat" : "Oynat");
  clearInterval(timer);
  if (!playing) return;
  const interval = Number(byId("playbackSpeed").value);
  timer = setInterval(() => {
    if (frameIndex >= replay.frames.length - 1) {
      playing = false;
      clearInterval(timer);
      byId("playIcon").textContent = "▶";
      return;
    }
    setFrame(frameIndex + 1);
  }, interval);
}

async function loadFile(file) {
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    replay = normalizeReplay(payload);
    frameIndex = 0;
    playing = false;
    clearInterval(timer);
    byId("playIcon").textContent = "▶";
  byId("replayStatus").textContent = "REPLAY";
    render();
  } catch (error) {
    window.alert(`Replay açılamadı: ${error.message}`);
  }
}

function formatNumber(value) {
  return new Intl.NumberFormat("tr-TR", { notation: Number(value) >= 1000000 ? "compact" : "standard" }).format(value);
}

function formatPercent(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function formatSigned(value) {
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

byId("timeline").addEventListener("input", (event) => setFrame(event.target.value));
byId("previousFrame").addEventListener("click", () => setFrame(frameIndex - 1));
byId("nextFrame").addEventListener("click", () => setFrame(frameIndex + 1));
byId("playPause").addEventListener("click", togglePlay);
byId("playbackSpeed").addEventListener("change", () => {
  if (playing) {
    togglePlay();
    togglePlay();
  }
});
byId("replayFile").addEventListener("change", (event) => loadFile(event.target.files[0]));
byId("spectatorToggle").addEventListener("click", (event) => {
  spectator = !spectator;
  event.currentTarget.setAttribute("aria-pressed", String(spectator));
  event.currentTarget.lastChild.textContent = spectator ? " Seyirci" : " Oyuncu";
  render();
});

const dropZone = byId("dropZone");
["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") byId("replayFile").click();
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select")) return;
  if (event.key === "ArrowLeft") setFrame(frameIndex - 1);
  if (event.key === "ArrowRight") setFrame(frameIndex + 1);
  if (event.key === " ") {
    event.preventDefault();
    togglePlay();
  }
});

render();
