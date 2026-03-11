/* ── AFL Match Predictor — Frontend ──────────────────────────────── */

const API = '';  // same origin

// ── Team data ───────────────────────────────────────────────────────

const TEAM_COLORS = {
  'Adelaide': '#002B5C',
  'Brisbane': '#7B2242',
  'Carlton': '#162749',
  'Collingwood': '#000000',
  'Essendon': '#CC2031',
  'Fremantle': '#2A0D54',
  'Geelong': '#1C3C63',
  'Gold Coast': '#D63239',
  'Greater Western Sydney': '#F15C22',
  'Hawthorn': '#4D2004',
  'Melbourne': '#0F1E3C',
  'North Melbourne': '#013B9F',
  'Port Adelaide': '#008AAB',
  'Richmond': '#FED102',
  'St Kilda': '#ED1C24',
  'Sydney': '#E2252B',
  'West Coast': '#062EE2',
  'Western Bulldogs': '#014896',
};

const TEAM_ABBREV = {
  'Adelaide': 'ADE',
  'Brisbane': 'BRI',
  'Carlton': 'CAR',
  'Collingwood': 'COL',
  'Essendon': 'ESS',
  'Fremantle': 'FRE',
  'Geelong': 'GEE',
  'Gold Coast': 'GCS',
  'Greater Western Sydney': 'GWS',
  'Hawthorn': 'HAW',
  'Melbourne': 'MEL',
  'North Melbourne': 'NTH',
  'Port Adelaide': 'PTA',
  'Richmond': 'RIC',
  'St Kilda': 'STK',
  'Sydney': 'SYD',
  'West Coast': 'WCE',
  'Western Bulldogs': 'WBD',
};

const TEAM_LOGO_URLS = {
  'Adelaide': 'https://squiggle.com.au/wp-content/uploads/2019/01/adelaide.png',
  'Brisbane': 'https://squiggle.com.au/wp-content/uploads/2019/01/brisbane.png',
  'Carlton': 'https://squiggle.com.au/wp-content/uploads/2019/01/carlton.png',
  'Collingwood': 'https://squiggle.com.au/wp-content/uploads/2019/01/collingwood.png',
  'Essendon': 'https://squiggle.com.au/wp-content/uploads/2019/01/essendon.png',
  'Fremantle': 'https://squiggle.com.au/wp-content/uploads/2019/01/fremantle.png',
  'Geelong': 'https://squiggle.com.au/wp-content/uploads/2019/01/geelong.png',
  'Gold Coast': 'https://squiggle.com.au/wp-content/uploads/2019/01/goldcoast.png',
  'Greater Western Sydney': 'https://squiggle.com.au/wp-content/uploads/2019/01/gws.png',
  'Hawthorn': 'https://squiggle.com.au/wp-content/uploads/2019/01/hawthorn.png',
  'Melbourne': 'https://squiggle.com.au/wp-content/uploads/2019/01/melbourne.png',
  'North Melbourne': 'https://squiggle.com.au/wp-content/uploads/2019/01/north.png',
  'Port Adelaide': 'https://squiggle.com.au/wp-content/uploads/2019/01/portadelaide.png',
  'Richmond': 'https://squiggle.com.au/wp-content/uploads/2019/01/richmond.png',
  'St Kilda': 'https://squiggle.com.au/wp-content/uploads/2019/01/stkilda.png',
  'Sydney': 'https://squiggle.com.au/wp-content/uploads/2019/01/sydney.png',
  'West Coast': 'https://squiggle.com.au/wp-content/uploads/2019/01/westcoast.png',
  'Western Bulldogs': 'https://squiggle.com.au/wp-content/uploads/2019/01/bulldogs.png',
};

function teamLogo(team, size = 56) {
  const url = TEAM_LOGO_URLS[team];
  const color = TEAM_COLORS[team] || '#333';
  const abbrev = TEAM_ABBREV[team] || team.substring(0, 3).toUpperCase();

  if (url) {
    return `<img src="${url}" alt="${team}"
      style="width:${size}px;height:${size}px;border-radius:50%;object-fit:contain;background:${color};padding:4px;border:2px solid rgba(255,255,255,0.1);"
      onerror="this.outerHTML='<div class=\\'team-logo\\' style=\\'width:${size}px;height:${size}px;background:${color}\\'>${abbrev}</div>'">`;
  }
  return `<div class="team-logo" style="width:${size}px;height:${size}px;background:${color}">${abbrev}</div>`;
}

// ── Navigation ──────────────────────────────────────────────────────

function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));

  const pageEl = document.getElementById(`page-${page}`);
  const navEl = document.querySelector(`[data-page="${page}"]`);

  if (pageEl) pageEl.classList.add('active');
  if (navEl) navEl.classList.add('active');

  // Load data for the page
  if (page === 'predictions') loadPredictions();
  if (page === 'elo') loadEloLadder();
  if (page === 'ladder') loadLadderPage();
  if (page === 'performance') loadPerformance();
}

// ── Predictions page ────────────────────────────────────────────────

let fixtureData = null;

async function loadPredictions() {
  const container = document.getElementById('predictions-content');
  container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading fixture...</div>';

  try {
    const res = await fetch(`${API}/fixture/2026`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    fixtureData = await res.json();

    renderRoundSelector();
    const upcomingRounds = [...new Set(fixtureData.upcoming.map(g => g.round_number))].sort((a, b) => a - b);
    if (upcomingRounds.length > 0) {
      loadRound(upcomingRounds[0]);
    } else {
      container.innerHTML = '<div class="empty-state"><p>No upcoming matches. The season may be complete.</p></div>';
    }
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Failed to load fixture: ${err.message}</div>`;
  }
}

function renderRoundSelector() {
  const selector = document.getElementById('round-selector');
  if (!fixtureData) return;

  const allRounds = [...new Set([
    ...fixtureData.upcoming.map(g => g.round_number),
  ])].sort((a, b) => a - b);

  const playedRounds = [...new Set(fixtureData.played.map(g => g.round_number))].sort((a, b) => a - b);

  let html = '<div class="round-selector"><label>Round:</label><select id="round-select" onchange="loadRound(parseInt(this.value))">';
  html += '<optgroup label="Upcoming">';
  for (const r of allRounds) {
    html += `<option value="${r}">${r === 0 ? 'Opening Round' : 'Round ' + r}</option>`;
  }
  html += '</optgroup>';
  if (playedRounds.length > 0) {
    html += '<optgroup label="Completed">';
    for (const r of playedRounds) {
      html += `<option value="${r}">${r === 0 ? 'Opening Round' : 'Round ' + r} (Played)</option>`;
    }
    html += '</optgroup>';
  }
  html += '</select></div>';
  selector.innerHTML = html;
}

async function loadRound(roundNum) {
  const container = document.getElementById('matches-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading predictions...</div>';

  // Check if this round has been played
  const playedGames = fixtureData?.played?.filter(g => g.round_number === roundNum) || [];
  if (playedGames.length > 0) {
    renderPlayedRound(playedGames, roundNum);
    return;
  }

  try {
    const res = await fetch(`${API}/round/2026/${roundNum}/predictions`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const data = await res.json();
    renderPredictions(data.predictions, roundNum);
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Failed to load predictions: ${err.message}</div>`;
  }
}

function renderPlayedRound(games, roundNum) {
  const container = document.getElementById('matches-container');
  let html = '<div class="matches-grid">';
  for (const g of games) {
    const homeWon = g.home_score > g.away_score;
    const awayWon = g.away_score > g.home_score;
    html += `
      <div class="match-card">
        <div class="match-header">
          <span class="match-venue">${g.venue}</span>
          <span class="match-confidence confidence-high">FINAL</span>
        </div>
        <div class="match-teams">
          <div class="team">
            ${teamLogo(g.home_team)}
            <span class="team-name">${g.home_team}</span>
            <span class="team-prob ${homeWon ? 'favoured' : 'underdog'}">${g.home_score}</span>
          </div>
          <span class="match-vs">vs</span>
          <div class="team">
            ${teamLogo(g.away_team)}
            <span class="team-name">${g.away_team}</span>
            <span class="team-prob ${awayWon ? 'favoured' : 'underdog'}">${g.away_score}</span>
          </div>
        </div>
        <div class="match-margin">${homeWon ? g.home_team : g.away_team} by ${Math.abs(g.home_score - g.away_score)} points</div>
      </div>`;
  }
  html += '</div>';
  container.innerHTML = html;
}

function renderPredictions(predictions, roundNum) {
  const container = document.getElementById('matches-container');

  if (!predictions || predictions.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No predictions available for this round.</p></div>';
    return;
  }

  let html = '<div class="matches-grid">';
  for (const p of predictions) {
    const homeProb = p.home_prob || 0.5;
    const awayProb = p.away_prob || 0.5;
    const homePct = Math.round(homeProb * 100);
    const awayPct = Math.round(awayProb * 100);
    const confidence = p.confidence || 'low';
    const margin = p.predicted_margin;
    const homeColor = TEAM_COLORS[p.home_team] || '#3b82f6';
    const awayColor = TEAM_COLORS[p.away_team] || '#ef4444';
    const favoured = homeProb >= 0.5 ? p.home_team : p.away_team;
    const marginAbs = Math.abs(margin).toFixed(0);

    html += `
      <div class="match-card">
        <div class="match-header">
          <span class="match-venue">${p.venue}</span>
          <span class="match-confidence confidence-${confidence}">${confidence}</span>
        </div>
        <div class="match-teams">
          <div class="team">
            ${teamLogo(p.home_team)}
            <span class="team-name">${p.home_team}</span>
            <span class="team-prob ${homeProb >= 0.5 ? 'favoured' : 'underdog'}">${homePct}%</span>
          </div>
          <span class="match-vs">vs</span>
          <div class="team">
            ${teamLogo(p.away_team)}
            <span class="team-name">${p.away_team}</span>
            <span class="team-prob ${awayProb > 0.5 ? 'favoured' : 'underdog'}">${awayPct}%</span>
          </div>
        </div>
        <div class="prob-bar-container">
          <div class="prob-bar-home" style="width:${homePct}%;background:${homeColor}"></div>
          <div class="prob-bar-away" style="width:${awayPct}%;background:${awayColor}"></div>
        </div>
        <div class="match-margin">${favoured} by ~${marginAbs} points</div>
      </div>`;
  }
  html += '</div>';
  container.innerHTML = html;
}

// ── ELO Ratings page ────────────────────────────────────────────────

async function loadEloLadder() {
  const container = document.getElementById('elo-content');
  container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading ELO ratings...</div>';

  try {
    const res = await fetch(`${API}/elo/ladder`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const teams = await res.json();
    renderEloLadder(teams);
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Failed to load ELO ratings: ${err.message}</div>`;
  }
}

function renderEloLadder(teams) {
  const container = document.getElementById('elo-content');

  if (!teams || teams.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No ELO data available.</p></div>';
    return;
  }

  const maxElo = Math.max(...teams.map(t => t.elo));
  const minElo = Math.min(...teams.map(t => t.elo));
  const range = maxElo - minElo || 1;

  let html = '<div class="elo-grid">';

  for (const team of teams) {
    const color = TEAM_COLORS[team.team] || '#3b82f6';
    const diffSign = team.diff_from_avg >= 0 ? '+' : '';
    const diffClass = team.diff_from_avg >= 0 ? 'positive' : 'negative';
    const barWidth = ((team.elo - minElo) / range * 70 + 30).toFixed(0);

    html += `
      <div class="elo-card">
        <div class="elo-rank">${team.rank}</div>
        <div class="elo-team-info">
          ${teamLogo(team.team, 44)}
          <div class="elo-team-details">
            <span class="elo-team-name">${team.team}</span>
            <div class="elo-bar-container">
              <div class="elo-bar" style="width:${barWidth}%;background:${color};"></div>
            </div>
          </div>
        </div>
        <div class="elo-values">
          <span class="elo-rating">${team.elo.toFixed(0)}</span>
          <span class="elo-diff ${diffClass}">${diffSign}${team.diff_from_avg.toFixed(0)}</span>
        </div>
      </div>`;
  }

  html += '</div>';
  html += '<p style="color:var(--text-muted);margin-top:1rem;font-size:0.8rem;">ELO ratings computed from 2015-2025 match results. Teams start at 1500 each season with 20% regression toward the mean. +/- shows distance from league average (1500).</p>';

  container.innerHTML = html;
}


// ── Ladder page ─────────────────────────────────────────────────────

async function loadLadderPage() {
  // Just show the controls; simulation runs on button click
}

async function runSimulation() {
  const btn = document.getElementById('sim-btn');
  const container = document.getElementById('ladder-content');
  const nSims = parseInt(document.getElementById('sim-count').value) || 1000;

  btn.disabled = true;
  btn.textContent = 'Simulating...';
  container.innerHTML = `<div class="loading"><div class="spinner"></div>Running ${nSims.toLocaleString()} simulations...</div>`;

  try {
    const res = await fetch(`${API}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_simulations: nSims }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `API error: ${res.status}`);
    }

    const data = await res.json();
    renderLadder(data);
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Simulation failed: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Simulation';
  }
}

function renderLadder(data) {
  const container = document.getElementById('ladder-content');
  const ladder = data.deterministic_ladder || [];
  const finalsProb = data.finals_probability || {};
  const flagProb = data.premiership_probability || {};

  if (ladder.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No simulation results.</p></div>';
    return;
  }

  let html = `
    <p style="color:var(--text-secondary);margin-bottom:1rem;font-size:0.85rem;">
      Based on ${data.n_simulations?.toLocaleString()} simulations |
      ${data.played_games} played | ${data.upcoming_games} remaining
    </p>
    <table class="ladder-table">
      <thead>
        <tr>
          <th class="num">#</th>
          <th>Team</th>
          <th class="num">W</th>
          <th class="num">L</th>
          <th class="num">D</th>
          <th class="num">Pts</th>
          <th class="num">%</th>
          <th class="num">Avg W</th>
          <th class="num">Finals %</th>
          <th class="num">Flag %</th>
        </tr>
      </thead>
      <tbody>`;

  ladder.forEach((team, i) => {
    const rank = i + 1;
    const isFinalsZone = rank <= 8;
    const isCutoff = rank === 8;
    const fp = team.finals_prob ?? finalsProb[team.team] ?? 0;
    const pp = team.flag_prob ?? flagProb[team.team] ?? 0;

    let fpClass = 'low';
    if (fp > 80) fpClass = 'high';
    else if (fp > 40) fpClass = 'medium';

    html += `
      <tr class="${isFinalsZone ? 'finals-zone' : ''} ${isCutoff ? 'finals-cutoff' : ''}">
        <td class="num" style="font-weight:700;color:var(--text-muted)">${rank}</td>
        <td>
          <div class="ladder-team-cell">
            ${teamLogo(team.team, 32)}
            <span class="ladder-team-name">${team.team}</span>
          </div>
        </td>
        <td class="num">${team.wins}</td>
        <td class="num">${team.losses}</td>
        <td class="num">${team.draws}</td>
        <td class="num" style="font-weight:700">${team.pts}</td>
        <td class="num">${team.percentage?.toFixed(1)}</td>
        <td class="num">${team.avg_wins?.toFixed(1) ?? '-'}</td>
        <td class="num"><span class="prob-badge ${fpClass}">${fp.toFixed(0)}%</span></td>
        <td class="num">${pp > 0 ? pp.toFixed(1) + '%' : '-'}</td>
      </tr>`;
  });

  html += '</tbody></table>';
  html += '<p style="color:var(--text-muted);margin-top:0.75rem;font-size:0.8rem;">Top 8 highlighted for finals qualification. Flag % = probability of finishing 1st (minor premiership).</p>';

  container.innerHTML = html;
}

function updateSimCount(val) {
  document.getElementById('sim-count-display').textContent = parseInt(val).toLocaleString();
}

// ── Performance page ────────────────────────────────────────────────

async function loadPerformance() {
  const container = document.getElementById('performance-content');
  container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading model metrics...</div>';

  try {
    const [infoRes, featRes] = await Promise.all([
      fetch(`${API}/model/info`),
      fetch(`${API}/model/features`),
    ]);

    let info = null, features = null;
    if (infoRes.ok) info = await infoRes.json();
    if (featRes.ok) features = await featRes.json();

    renderPerformance(info, features);
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Failed to load metrics: ${err.message}</div>`;
  }
}

function renderPerformance(info, features) {
  const container = document.getElementById('performance-content');

  let html = '';

  if (info) {
    html += `
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Accuracy</div>
          <div class="metric-value">${(info.accuracy * 100).toFixed(1)}%</div>
          <div class="metric-detail">+8.7pp vs baseline</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Log Loss</div>
          <div class="metric-value">${info.log_loss.toFixed(3)}</div>
          <div class="metric-detail">vs 0.693 random</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Features</div>
          <div class="metric-value">${info.feature_count}</div>
          <div class="metric-detail">Engineered features</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Model</div>
          <div class="metric-value" style="font-size:1rem;margin-top:8px">XGB + LGBM</div>
          <div class="metric-detail">Platt-calibrated ensemble</div>
        </div>
      </div>`;
  }

  // Model comparison
  html += `
    <div class="section-header" style="margin-top:2rem;">
      <h2>Model Comparison</h2>
      <p>Performance on the held-out temporal test set (2024+ games)</p>
    </div>
    <div class="model-comparison">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Log Loss</th>
            <th>Brier Score</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>XGBoost</td><td>64.7%</td><td>0.600</td><td>0.210</td></tr>
          <tr><td>LightGBM</td><td>65.2%</td><td>0.577</td><td>0.199</td></tr>
          <tr class="best"><td>Ensemble (avg)</td><td>65.7%</td><td>0.585</td><td>0.203</td></tr>
          <tr><td>Calibrated</td><td>65.0%</td><td>0.602</td><td>0.208</td></tr>
        </tbody>
      </table>
    </div>`;

  // Feature importance
  if (features && features.length > 0) {
    html += `
      <div class="section-header" style="margin-top:2rem;">
        <h2>Feature Importance</h2>
        <p>Top features driving predictions (XGBoost gain)</p>
      </div>`;

    const top10 = features.slice(0, 10);
    const maxImp = top10[0]?.contribution || 1;

    html += '<div style="margin-top:1rem;">';
    for (const f of top10) {
      const pct = (f.contribution / maxImp * 100).toFixed(0);
      const displayName = f.feature
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());

      html += `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <span style="width:160px;font-size:0.85rem;color:var(--text-secondary);text-align:right;">${displayName}</span>
          <div style="flex:1;height:24px;background:var(--bg-card);border-radius:4px;overflow:hidden;">
            <div style="width:${pct}%;height:100%;background:linear-gradient(90deg,var(--accent),#818cf8);border-radius:4px;transition:width 0.5s;"></div>
          </div>
          <span style="width:50px;font-size:0.8rem;color:var(--text-muted);">${f.contribution.toFixed(3)}</span>
        </div>`;
    }
    html += '</div>';
  }

  container.innerHTML = html;
}

// ── Init ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  navigate('predictions');
});
