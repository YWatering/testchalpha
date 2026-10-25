const state = { result: null, horizons: [1, 5, 10, 20], chartSize: 64, sourceHash: null };

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[char]));
const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const number = Number(value) * 100;
  return `${number >= 0 ? '+' : ''}${number.toFixed(2)}%`;
};
const formatDistance = (value) => Number(value).toFixed(4);
const chartLabel = { monthly: '月线 · 20D/根', weekly: '周线 · 5D/根', daily: '日线 · 1D/根', future: '未来 · 20D' };

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || '请求失败');
  return payload;
}

function setNotice(message, tone = '') {
  const notice = $('#notice');
  notice.textContent = message;
  notice.className = `notice ${tone}`.trim();
}

function setImageCard(name, src) {
  const dimension = `${state.chartSize} × ${state.chartSize}`;
  return `<article class="chart-card"><div class="chart-card-header"><strong>${chartLabel[name] || name}</strong><span data-chart-dimension>${dimension}</span></div><div class="chart-frame"><img src="${src}" alt="${chartLabel[name] || name} K 线图" /></div></article>`;
}

function renderQuery(query) {
  $('#query-title').textContent = `${query.symbol} · ${query.date}`;
  $('#query-meta').textContent = '历史窗口已构建 · 未来图不参与检索';
  $('#query-charts').innerHTML = ['monthly', 'weekly', 'daily']
    .map((name) => setImageCard(name, query.charts[name])).join('');
}

function returnClass(value) {
  if (value === null || value === undefined) return '';
  return Number(value) >= 0 ? 'positive' : 'negative';
}

function renderStats(horizon) {
  if (!state.result) return;
  const stat = state.result.statistics[String(horizon)];
  if (!stat) return;
  const cards = [
    ['加权平均收益', formatPercent(stat.weighted_mean)],
    ['简单平均收益', formatPercent(stat.mean)],
    ['加权标准差', formatPercent(stat.weighted_std)],
    ['稳定性得分', stat.signal_strength === null || !Number.isFinite(Number(stat.signal_strength)) ? '—' : Number(stat.signal_strength).toFixed(2)],
    ['有效样本数', stat.effective_sample_size === null ? '—' : Number(stat.effective_sample_size).toFixed(1)],
    ['95%区间下界', formatPercent(stat.lower_bound)],
  ];
  $('#stats-grid').innerHTML = cards.map(([label, value], index) => `<div class="stat-card"><span>${label}</span><strong class="${returnClass(index === 0 ? stat.weighted_mean : stat.mean)}">${value}</strong></div>`).join('');
  $('#distribution-caption').textContent = `未来 ${horizon} 个交易日 · ${stat.count} 个历史样本`;
  renderDistribution(stat);
}

function renderDistribution(stat) {
  const container = $('#distribution-chart');
  const edges = stat.histogram.edges || [];
  const counts = stat.histogram.counts || [];
  const kdeX = stat.kde.x || [];
  const kdeY = stat.kde.density || [];
  if (!counts.length) {
    container.innerHTML = '<div class="empty-state">没有可用的收益分布数据。</div>';
    return;
  }
  const width = 760; const height = 250; const left = 42; const right = 16; const top = 16; const bottom = 32;
  const plotWidth = width - left - right; const plotHeight = height - top - bottom;
  const maxCount = Math.max(...counts, 1); const maxDensity = Math.max(...kdeY, 1e-9);
  const minX = edges[0]; const maxX = edges[edges.length - 1];
  const xScale = (value) => left + ((value - minX) / (maxX - minX || 1)) * plotWidth;
  const yCount = (value) => top + plotHeight - (value / maxCount) * plotHeight;
  const yDensity = (value) => top + plotHeight - (value / maxDensity) * plotHeight;
  const bars = counts.map((count, index) => {
    const x = xScale(edges[index]) + 1;
    const next = xScale(edges[index + 1]) - 1;
    const barWidth = Math.max(1, next - x);
    return `<rect class="bar" x="${x}" y="${yCount(count)}" width="${barWidth}" height="${top + plotHeight - yCount(count)}" />`;
  }).join('');
  const path = kdeX.map((value, index) => `${index ? 'L' : 'M'} ${xScale(value).toFixed(2)} ${yDensity(kdeY[index]).toFixed(2)}`).join(' ');
  const zero = minX <= 0 && maxX >= 0 ? `<line class="axis-line" x1="${xScale(0)}" y1="${top}" x2="${xScale(0)}" y2="${top + plotHeight}" />` : '';
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="收益直方图和 KDE 曲线"><line class="axis-line" x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}" />${zero}${bars}<path class="kde-line" d="${path}" /><text class="axis-label" x="${left}" y="${height - 8}">${formatPercent(minX)}</text><text class="axis-label" text-anchor="end" x="${width - right}" y="${height - 8}">${formatPercent(maxX)}</text><text class="axis-label" x="${left}" y="${top - 2}">${maxCount} 样本</text></svg>`;
}

function renderResults(result) {
  const matches = result.matches || [];
  $('#result-count').textContent = `${matches.length} 个样本`;
  if (!matches.length) {
    $('#results-table-body').innerHTML = '<tr><td colspan="7" class="table-empty">没有符合条件的历史样本。</td></tr>';
    $('#result-cards').innerHTML = '';
    return;
  }
  $('#results-table-body').innerHTML = matches.map((match) => {
    const returns = match.returns || {};
    return `<tr><td class="rank-cell">${match.rank}</td><td class="sample-cell"><strong>${escapeHtml(match.symbol)}</strong><span>${escapeHtml(match.date)}</span></td><td class="distance-cell">${formatDistance(match.distance)}</td>${state.horizons.map((horizon) => `<td class="${returnClass(returns[String(horizon)])}">${formatPercent(returns[String(horizon)])}</td>`).join('')}</tr>`;
  }).join('');
  $('#result-cards').innerHTML = matches.map((match) => {
    const returns = match.returns || {};
    const chartNames = ['monthly', 'weekly', 'daily'];
    const charts = chartNames.map((name) => `<div class="mini-chart"><span>${chartLabel[name]}</span><img src="${match.charts[name]}" alt="${chartLabel[name]}" /></div>`).join('');
    const allCharts = charts + `<div class="mini-chart"><span>${chartLabel.future}</span><img src="${match.future_chart}" alt="未来 20 日走势" /></div>`;
    return `<article class="match-card"><div class="match-card-top"><div class="match-card-title">#${match.rank} · ${escapeHtml(match.symbol)} <span class="match-card-distance">${escapeHtml(match.date)} · 距离 ${formatDistance(match.distance)} · 权重 ${(Number(match.similarity_weight || 0) * 100).toFixed(1)}%</span></div></div><div class="match-chart-grid">${allCharts}</div><div class="match-return-row">${state.horizons.map((horizon) => `<span>未来 ${horizon}D <strong class="${returnClass(returns[String(horizon)])}">${formatPercent(returns[String(horizon)])}</strong></span>`).join('')}</div></article>`;
  }).join('');
}

function updateWeights() {
  $('#month-weight-value').textContent = Number($('#month-weight').value).toFixed(2);
  $('#week-weight-value').textContent = Number($('#week-weight').value).toFixed(2);
}

async function loadDates(symbol, keepCurrent = false) {
  const payload = await api(`/api/dates?symbol=${encodeURIComponent(symbol)}`);
  const dates = payload.dates || [];
  const dateInput = $('#date-input');
  dateInput.min = dates[0] || '';
  dateInput.max = dates[dates.length - 1] || '';
  if (!keepCurrent || !dates.includes(dateInput.value)) dateInput.value = dates[dates.length - 1] || '';
}

async function bootstrap() {
  try {
    const [health, symbols, config] = await Promise.all([api('/api/health'), api('/api/symbols'), api('/api/retrieval/config')]);
    $('#health-label').textContent = '数据服务在线';
    $('.status-dot').classList.add('online');
    const dataset = health.dataset || {};
    state.sourceHash = health.source_hash;
    const endpointName = dataset.endpoint === 'sina' ? 'Sina' : dataset.endpoint === 'eastmoney' ? 'Eastmoney' : '';
    const sourceName = dataset.provider === 'akshare' ? `AKShare${endpointName ? ` · ${endpointName}` : ''}` : dataset.provider === 'synthetic' ? '合成演示' : '本地文件';
    const adjustName = dataset.adjust ? ` · ${dataset.adjust}` : '';
    const storageName = health.storage === 'sqlite' ? ' · SQLite 自动刷新' : '';
    const indexName = health.index?.mode === 'offline_colproj' ? ' · ColProj 离线索引' : '';
    $('#data-chip').textContent = `${health.symbols} 只股票 · ${sourceName}${adjustName}${storageName}${indexName}`;
    $('#data-chip').title = dataset.data_start_date && dataset.data_end_date
      ? `行情区间 ${dataset.data_start_date} 至 ${dataset.data_end_date} · SHA-256 ${dataset.sha256 || health.source_hash}`
      : `数据文件 SHA-256 ${health.source_hash}`;
    state.horizons = config.future_horizons || state.horizons;
    state.chartSize = Number((config.chart_size || [64])[0]);
    $('#horizon-input').innerHTML = state.horizons.map((horizon) => `<option value="${horizon}">未来 ${horizon}D</option>`).join('');
    const symbolInput = $('#symbol-input');
    const names = Object.fromEntries((symbols.instruments || []).map((item) => [item.symbol, item.name]));
    symbolInput.innerHTML = symbols.symbols.map((symbol) => {
      const name = names[symbol];
      const label = name && name !== symbol ? `${symbol}·${name}` : symbol;
      return `<option value="${escapeHtml(symbol)}">${escapeHtml(label)}</option>`;
    }).join('');
    await loadDates(symbolInput.value);
    setNotice('选择参数后执行检索。');
  } catch (error) {
    $('#health-label').textContent = '数据服务异常';
    setNotice(error.message, 'error');
  }
}

async function runSearch(event) {
  event.preventDefault();
  const button = $('#search-button');
  button.disabled = true;
  button.textContent = '检索中…';
  setNotice('正在生成查询图、扫描历史候选并计算未来收益。');
  try {
    const result = await api('/api/retrieval/search', {
      method: 'POST',
      body: JSON.stringify({
        symbol: $('#symbol-input').value,
        date: $('#date-input').value,
        model: $('#model-input').value,
        month_weight: Number($('#month-weight').value),
        week_weight: Number($('#week-weight').value),
        day_weight: 1,
        top_k: Number($('#topk-input').value),
        embargo_trading_days: Number($('#embargo-input').value),
      }),
    });
    state.result = result;
    renderQuery(result.query);
    renderResults(result);
    renderStats(Number($('#horizon-input').value));
    const perf = result.performance || {};
    setNotice(`检索完成：${result.matches.length}/${perf.candidate_count ?? '—'} 个候选 · 检索 ${perf.total_ms ?? '—'} ms / 请求 ${perf.request_ms ?? '—'} ms · 缓存命中 ${perf.cache_hits ?? 0}`, 'success');
  } catch (error) {
    setNotice(error.message, 'error');
  } finally {
    button.disabled = false;
    button.innerHTML = '开始检索 <span>→</span>';
  }
}

$('#symbol-input').addEventListener('change', () => loadDates($('#symbol-input').value).catch((error) => setNotice(error.message, 'error')));
$('#month-weight').addEventListener('input', updateWeights);
$('#week-weight').addEventListener('input', updateWeights);
$('#horizon-input').addEventListener('change', () => renderStats(Number($('#horizon-input').value)));
$('#search-form').addEventListener('submit', runSearch);
updateWeights();
bootstrap();
setInterval(async () => {
  try {
    const health = await api('/api/health');
    if (state.sourceHash && health.source_hash !== state.sourceHash) window.location.reload();
  } catch (_) {
    // The normal health state already reports persistent service failures.
  }
}, 60000);
