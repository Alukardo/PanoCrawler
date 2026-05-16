(function () {
  const dom = document.getElementById("globe");
  const chart = echarts.init(dom, null, {
    renderer: "canvas",
    useDirtyRect: false
  });

  const elements = {
    loadConfiguredData: document.getElementById("loadConfiguredData"),
    clearData: document.getElementById("clearData"),
    themeToggle: document.getElementById("themeToggle"),
    downloadCount: document.getElementById("downloadCount"),
    visibleCount: document.getElementById("visibleCount"),
    statusText: document.getElementById("statusText"),
    detailPanel: document.getElementById("detailPanel")
  };

  const textureData = window.PANO_TEXTURES || {};
  const ASSETS = {
    earth: textureData.earth || "assets/world.topo.bathy.200401.jpg",
    starfield: textureData.starfield || "assets/starfield.jpg"
  };
  const CONFIG_URL = "../config.yaml";
  const RUNTIME_METADATA_URL = "../runtime/info.csv";
  const PROJECT_IMAGES_METADATA_URL = "../images/info.csv";
  const SEARCH_POINT_COLOR = "#ff9f0a";
  const POINT_HOVER_COLOR = "#ffd166";
  const LEGACY_SEARCH_POINT_ID = "unknown";
  const CAMERA_MOVE_MS = 1800;
  const DEFAULT_CAMERA_DISTANCE = 160;
  const FOCUS_CAMERA_DISTANCE = 150;

  const state = {
    records: [],
    points: [],
    searchPoints: [],
    config: null,
    selected: null,
    cameraTarget: null,
    cameraAnimationId: null,
    optionReady: false
  };

  function setStatus(message) {
    elements.statusText.textContent = message;
  }

  function stripInlineComment(value) {
    let quoted = false;
    let quote = "";
    for (let i = 0; i < value.length; i += 1) {
      const char = value[i];
      if ((char === '"' || char === "'") && value[i - 1] !== "\\") {
        if (!quoted) {
          quoted = true;
          quote = char;
        } else if (quote === char) {
          quoted = false;
        }
      }
      if (char === "#" && !quoted) {
        return value.slice(0, i).trim();
      }
    }
    return value.trim();
  }

  function parseYamlValue(value) {
    const clean = stripInlineComment(value);
    if (!clean) return "";
    if ((clean.startsWith('"') && clean.endsWith('"')) || (clean.startsWith("'") && clean.endsWith("'"))) {
      return clean.slice(1, -1);
    }
    if (clean.startsWith("[") && clean.endsWith("]")) {
      return clean.slice(1, -1).split(",").map(item => parseYamlValue(item.trim())).filter(Boolean);
    }
    if (clean === "true") return true;
    if (clean === "false") return false;
    if (/^-?\d+(?:\.\d+)?$/.test(clean)) return Number(clean);
    return clean;
  }

  function parseSimpleYaml(text) {
    const config = {};
    for (const line of text.split(/\r?\n/)) {
      const match = line.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
      if (!match) continue;
      config[match[1]] = parseYamlValue(match[2]);
    }
    return config;
  }

  function configuredMetadataCandidates(config) {
    const candidates = [RUNTIME_METADATA_URL];
    const metadataPath = String(config.metadata_path || "info.csv");
    const imagesRoot = String(config.images_root || "images");
    if (!metadataPath.startsWith("/") && !imagesRoot.startsWith("/")) {
      candidates.push(`../${imagesRoot.replace(/\/$/, "")}/${metadataPath}`);
    }
    candidates.push(PROJECT_IMAGES_METADATA_URL);
    return Array.from(new Set(candidates));
  }

  async function fetchFirstText(urls) {
    const errors = [];
    for (const url of urls) {
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) throw new Error(`${response.status}`);
        return { url, text: await response.text() };
      } catch (error) {
        errors.push(`${url}: ${error.message}`);
      }
    }
    throw new Error(errors.join("; "));
  }

  function parseCsv(text) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      const next = text[i + 1];
      if (quoted) {
        if (char === '"' && next === '"') {
          field += '"';
          i += 1;
        } else if (char === '"') {
          quoted = false;
        } else {
          field += char;
        }
      } else if (char === '"') {
        quoted = true;
      } else if (char === ",") {
        row.push(field);
        field = "";
      } else if (char === "\n") {
        row.push(field);
        rows.push(row);
        row = [];
        field = "";
      } else if (char !== "\r") {
        field += char;
      }
    }
    if (field || row.length) {
      row.push(field);
      rows.push(row);
    }
    if (!rows.length) return [];
    const headers = rows.shift().map(value => value.trim());
    return rows
      .filter(values => values.some(value => String(value).trim()))
      .map(values => Object.fromEntries(headers.map((header, index) => [header, (values[index] || "").trim()])));
  }

  function normalizePoint(row, index) {
    const lat = Number(row.lat);
    const lon = Number(row.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || !row.pano_id) return null;
    return {
      type: "download",
      pano_id: row.pano_id,
      lat,
      lon,
      heading: row.heading || "",
      pitch: row.pitch || "",
      roll: row.roll || "",
      date: row.date || "",
      search_point: row.search_point || "",
      search_point_id: (row.search_point_id || LEGACY_SEARCH_POINT_ID).trim() || LEGACY_SEARCH_POINT_ID,
      order: index,
      metadata: row
    };
  }

  function parseSearchPoint(value) {
    const text = String(value || "").trim();
    const match = text.match(/^\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]$/);
    if (!match) return null;
    const lat = Number(match[1]);
    const lon = Number(match[2]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    return { lat, lon };
  }

  function searchCoordForPoint(point) {
    return parseSearchPoint(point.search_point) || { lat: point.lat, lon: point.lon };
  }

  function buildSearchPoints(points) {
    const groups = new Map();
    for (const point of points) {
      const coord = searchCoordForPoint(point);
      const key = `${coord.lat.toFixed(6)},${coord.lon.toFixed(6)}`;
      let group = groups.get(key);
      if (!group) {
        group = {
          type: "search",
          key,
          lat: coord.lat,
          lon: coord.lon,
          search_point: `[${coord.lat.toFixed(6)}, ${coord.lon.toFixed(6)}]`,
          // search_point_id 与 search_point 1-1，全体 downloads 共享。
          // 取首个非 legacy 值作为 group 身份。
          search_point_id: (point.search_point_id && point.search_point_id !== LEGACY_SEARCH_POINT_ID)
            ? point.search_point_id
            : LEGACY_SEARCH_POINT_ID,
          downloads: [],
          order: point.order
        };
        groups.set(key, group);
      } else if (group.search_point_id === LEGACY_SEARCH_POINT_ID
        && point.search_point_id && point.search_point_id !== LEGACY_SEARCH_POINT_ID) {
        // 升级 legacy 点: 后续遇到带真实 id 的 pano 就接手。
        group.search_point_id = point.search_point_id;
      }
      group.downloads.push(point);
      group.order = Math.min(group.order, point.order);
    }
    // 1-based displayIndex is human-friendly ("#3") and stable across reloads
    // because the sort key (earliest CSV row) is deterministic.
    return Array.from(groups.values())
      .sort((a, b) => a.order - b.order)
      .map((group, index) => Object.assign(group, { displayIndex: index + 1 }));
  }

  function updateData() {
    state.points = state.records.map(normalizePoint).filter(Boolean);
    state.points.sort((a, b) => {
      return a.order - b.order;
    });
    state.searchPoints = buildSearchPoints(state.points);
    state.selected = null;
    state.cameraTarget = null;
    stopCameraAnimation();
    updateSummary();
    updateDetail(null);
    renderGlobe({ reset: true });
  }

  function visiblePoints() {
    return state.searchPoints;
  }

  function stopCameraAnimation() {
    if (state.cameraAnimationId) {
      cancelAnimationFrame(state.cameraAnimationId);
      state.cameraAnimationId = null;
    }
  }

  function shortestLonDelta(fromLon, toLon) {
    return ((((toLon - fromLon) % 360) + 540) % 360) - 180;
  }

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function applyCameraTarget(coord, animate = false) {
    state.cameraTarget = coord;
    chart.setOption({
      globe: {
        viewControl: {
          autoRotate: false,
          distance: FOCUS_CAMERA_DISTANCE,
          minDistance: 80,
          maxDistance: 260,
          alpha: 24,
          beta: 158,
          targetCoord: coord,
          animation: animate,
          animationDurationUpdate: animate ? CAMERA_MOVE_MS : 0
        }
      }
    });
  }

  function initialCameraTarget() {
    return [0, 0];
  }

  function animateCameraToPoint(point) {
    if (!point || !Number.isFinite(point.lon) || !Number.isFinite(point.lat)) return;
    const to = [point.lon, point.lat];
    const from = state.cameraTarget || initialCameraTarget();
    const start = performance.now();
    const lonDelta = shortestLonDelta(from[0], to[0]);
    const latDelta = to[1] - from[1];
    stopCameraAnimation();
    const tick = now => {
      const progress = Math.min(1, (now - start) / CAMERA_MOVE_MS);
      const eased = easeInOutCubic(progress);
      const coord = [from[0] + lonDelta * eased, from[1] + latDelta * eased];
      applyCameraTarget(coord);
      if (progress < 1) {
        state.cameraAnimationId = requestAnimationFrame(tick);
      } else {
        applyCameraTarget(to);
        state.cameraAnimationId = null;
      }
    };
    state.cameraAnimationId = requestAnimationFrame(tick);
  }

  function searchPointStyle() {
    return {
      color: SEARCH_POINT_COLOR,
      opacity: 1,
      borderWidth: 0.01,
      borderColor: "rgba(255,255,255,0.92)"
    };
  }

  function dataItem(point, index, kind) {
    return {
      name: "搜索点",
      value: [Number(point.lon), Number(point.lat)],
      itemStyle: searchPointStyle(),
      searchKey: point.key,
      searchIndex: index,
      downloadCount: Array.isArray(point.downloads) ? point.downloads.length : 0,
      kind
    };
  }

  function pointFromParams(params) {
    if (!params) return null;
    const data = params.data || {};
    if (data.raw) return data.raw;
    const index = Number(data.searchIndex ?? params.dataIndex);
    if (Number.isInteger(index) && index >= 0) {
      return state.searchPoints[index] || null;
    }
    if (data.searchKey) {
      return state.searchPoints.find(point => point.key === data.searchKey) || null;
    }
    return null;
  }

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatCoord(value) {
    const number = finiteNumber(value);
    return number === null ? "-" : number.toFixed(6);
  }

  function tooltipCoord(point, value, key, index) {
    const rawValue = finiteNumber(point[key]);
    if (rawValue !== null) return rawValue;
    if (Array.isArray(value)) {
      const arrayValue = finiteNumber(value[index]);
      if (arrayValue !== null) return arrayValue;
    }
    return null;
  }

  function tooltipCardStyle() {
    const light = document.documentElement.dataset.theme === "light";
    return {
      trigger: "item",
      renderMode: "html",
      confine: true,
      backgroundColor: light ? "rgba(255, 255, 255, 0.88)" : "rgba(10, 16, 28, 0.9)",
      borderColor: light ? "rgba(20, 40, 80, 0.14)" : "rgba(255, 255, 255, 0.16)",
      borderWidth: 1,
      padding: 0,
      textStyle: {
        color: light ? "#0f172a" : "#f8fafc",
        fontSize: 12
      },
      extraCssText: "border-radius:16px;box-shadow:0 18px 44px rgba(0,0,0,0.24);backdrop-filter:blur(16px);overflow:hidden;"
    };
  }

  function baseOption(series) {
    return {
      backgroundColor: "#000",
      tooltip: {
        ...tooltipCardStyle(),
        formatter: params => tooltipText(params)
      },
      globe: {
        baseTexture: ASSETS.earth,
        heightTexture: ASSETS.earth,
        displacementScale: 0.006,
        shading: "realistic",
        environment: ASSETS.starfield,
        realisticMaterial: {
          roughness: 1,
          metalness: 0
        },
        postEffect: {
          enable: false
        },
        light: {
          main: {
            intensity: 1.6,
            shadow: false
          },
          ambient: {
            intensity: 1.05
          }
        },
        viewControl: {
          autoRotate: !state.cameraTarget,
          autoRotateAfterStill: 3,
          autoRotateSpeed: 2,
          distance: state.cameraTarget ? FOCUS_CAMERA_DISTANCE : DEFAULT_CAMERA_DISTANCE,
          minDistance: 80,
          maxDistance: 260,
          alpha: 24,
          beta: 158,
          targetCoord: state.cameraTarget || undefined,
          animation: true,
          animationDurationUpdate: 900
        }
      },
      series
    };
  }

  function searchKeyForDownload(downloadPoint) {
    const coord = searchCoordForPoint(downloadPoint);
    return `${coord.lat.toFixed(6)},${coord.lon.toFixed(6)}`;
  }

  function renderGlobe(options = {}) {
    const points = visiblePoints();
    const series = [
      {
        id: "search-points",
        name: "搜索点",
        type: "scatter3D",
        coordinateSystem: "globe",
        blendMode: "lighter",
        shading: "color",
        dimensions: ["lon", "lat"],
        silent: false,
        animation: false,
        animationDurationUpdate: 0,
        symbolSize: 5.6,
        data: points.map((point, index) => dataItem(point, index, "search")),
        itemStyle: searchPointStyle(),
        label: {
          show: false
        },
        tooltip: {
          ...tooltipCardStyle(),
          formatter: params => tooltipText(params)
        },
        emphasis: {
          itemStyle: {
            color: POINT_HOVER_COLOR,
            borderWidth: 0.8,
            borderColor: "#fff",
            opacity: 1
          },
          label: {
            show: false
          }
        }
      }
    ];
    if (options.reset || !state.optionReady) {
      // Full rewrite — used on initial load to seed the globe + series.
      chart.setOption(baseOption(series), true);
      state.optionReady = true;
    } else {
      // Merge mode keeps `globe` (the planet itself) and other root options;
      // ECharts merges series by `id`, so updating just the series here is
      // safe as long as we don't intend to *remove* any of them.
      chart.setOption({ series }, false, false);
    }
    elements.visibleCount.textContent = String(points.length);
  }

  function tooltipText(params) {
    const point = pointFromParams(params) || {};
    const data = params.data || {};
    const value = Array.isArray(data.value) ? data.value : params.value || [];
    const lon = tooltipCoord(point, value, "lon", 0);
    const lat = tooltipCoord(point, value, "lat", 1);
    const count = Array.isArray(point.downloads) ? point.downloads.length : 0;
    const location = `${formatCoord(lat)}, ${formatCoord(lon)}`;
    // search_point_id == anchor pano_id; 1-1 with this search point.
    // Show it as the search point's identity in place of the old #N label.
    const idLabel = point.search_point_id && point.search_point_id !== LEGACY_SEARCH_POINT_ID
      ? point.search_point_id
      : (Number.isInteger(point.displayIndex) ? `#${point.displayIndex}` : "—");
    return `
      <div class="search-tooltip-card">
        <div class="search-tooltip-title">搜索点</div>
        <div class="search-tooltip-row">
          <span>ID</span>
          <strong>${escapeHtml(idLabel)}</strong>
        </div>
        <div class="search-tooltip-row">
          <span>地点</span>
          <strong>${escapeHtml(location)}</strong>
        </div>
        <div class="search-tooltip-row">
          <span>下载点</span>
          <strong>${count}</strong>
        </div>
      </div>
    `;
  }

  function updateSummary() {
    const downloadCount = state.points.length || state.records.length;
    elements.downloadCount.textContent = String(downloadCount);
    elements.visibleCount.textContent = String(visiblePoints().length);
  }

  function updateDetail(point) {
    const target = point || state.selected;
    if (!target) {
      elements.detailPanel.innerHTML = `<p class="detail-empty">点击地球上的搜索点查看对应下载点列表。</p>`;
      return;
    }
    const idLabel = (target.search_point_id && target.search_point_id !== LEGACY_SEARCH_POINT_ID)
      ? target.search_point_id
      : (Number.isInteger(target.displayIndex) ? `#${target.displayIndex}` : "");
    const titleSuffix = idLabel ? ` ${escapeHtml(idLabel)}` : "";
    elements.detailPanel.innerHTML = `
      <div class="panel-title-row">
        <div>
          <p class="detail-type">Downloaded Panoramas</p>
          <h2 class="detail-name">下载点列表${titleSuffix}</h2>
        </div>
      </div>
      ${downloadCards(target.downloads || [])}
    `;
  }

  function detailItem(label, value) {
    const displayValue = value === null || value === undefined || value === "" ? "-" : value;
    return `<div class="detail-item"><span>${escapeHtml(String(label))}</span><span>${escapeHtml(String(displayValue))}</span></div>`;
  }

  function downloadCards(downloads) {
    if (!downloads.length) return `<p class="detail-empty">没有对应下载点。</p>`;
    return `
      <div class="download-card-list">
        ${downloads.map(downloadCard).join("")}
      </div>
    `;
  }

  function downloadCard(point, index) {
    const idDisplay = (!point.search_point_id || point.search_point_id === LEGACY_SEARCH_POINT_ID)
      ? "—"
      : point.search_point_id;
    const rows = [
      { label: "Pano ID", value: point.pano_id || "-" },
      { label: "地点", value: `${formatCoord(point.lat)}, ${formatCoord(point.lon)}` },
      { label: "日期", value: point.date || "当前" },
      { label: "Heading", value: point.heading || "-" },
      { label: "Pitch", value: point.pitch || "-" },
      { label: "Roll", value: point.roll || "-" },
      { label: "Search Point ID", value: idDisplay }
    ];
    return `
      <article class="download-card">
        <div class="download-card-grid">
          ${rows.map(row => detailItem(row.label, row.value)).join("")}
        </div>
      </article>
    `;
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
  }

  async function loadConfiguredData() {
    if (window.location.protocol === "file:") {
      setStatus("自动读取数据需要使用项目根目录 HTTP 服务打开。");
      return;
    }

    elements.loadConfiguredData.disabled = true;
    setStatus("正在读取数据...");
    try {
      const configResponse = await fetch(CONFIG_URL, { cache: "no-store" });
      if (!configResponse.ok) throw new Error(`无法读取 ${CONFIG_URL}`);
      const configText = await configResponse.text();
      const config = parseSimpleYaml(configText);
      const metadata = await fetchFirstText(configuredMetadataCandidates(config));

      state.config = config;
      state.records = parseCsv(metadata.text);
      updateData();
      setStatus(`已从 ${metadata.url} 读取 ${state.points.length} 条全景记录，聚合为 ${state.searchPoints.length} 个搜索点。`);
    } catch (error) {
      setStatus(`读取数据失败：${error.message}。请确认从项目根目录启动 HTTP 服务，并确保数据文件可访问。`);
    } finally {
      elements.loadConfiguredData.disabled = false;
    }
  }

  function clearData() {
    state.records = [];
    state.points = [];
    state.searchPoints = [];
    state.config = null;
    state.selected = null;
    state.cameraTarget = null;
    stopCameraAnimation();
    updateSummary();
    updateDetail(null);
    renderGlobe({ reset: true });
    setStatus("已清空数据。请读取配置与数据。");
  }

  function toggleTheme() {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("pano-visual-theme", next);
    elements.themeToggle.textContent = next === "light" ? "深色" : "浅色";
    renderGlobe({ reset: true });
  }

  function bindEvents() {
    elements.loadConfiguredData.addEventListener("click", loadConfiguredData);
    elements.clearData.addEventListener("click", clearData);
    elements.themeToggle.addEventListener("click", toggleTheme);
    chart.on("click", params => {
      const point = pointFromParams(params);
      if (!point) return;
      state.selected = point;
      updateDetail(point);
      renderGlobe();
      animateCameraToPoint(point);
    });
    window.addEventListener("resize", () => chart.resize());
  }

  function initTheme() {
    const stored = localStorage.getItem("pano-visual-theme");
    const preferredLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    const theme = stored || (preferredLight ? "light" : "dark");
    document.documentElement.dataset.theme = theme;
    elements.themeToggle.textContent = theme === "light" ? "深色" : "浅色";
  }

  function init() {
    initTheme();
    bindEvents();
    updateSummary();
    updateDetail(null);
    renderGlobe({ reset: true });
    if (window.location.protocol === "file:") {
      setStatus("准备就绪。自动读取数据需要使用项目根目录 HTTP 服务打开。");
    } else {
      setStatus("准备就绪，正在自动读取配置与数据...");
      loadConfiguredData();
    }
  }

  init();
})();
