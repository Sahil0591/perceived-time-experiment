/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run "npm run dev" in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run "npm run deploy" to publish your worker
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

export default {
  async fetch(req, env) {
    const TZ = env.REPORT_TZ || 'Europe/London';
    function dayInTZ(d, tz = 'Europe/London', sep = '-') {
      const parts = new Intl.DateTimeFormat('en-GB', {
        timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit'
      }).formatToParts(d).reduce((acc, p) => {
        if (p.type !== 'literal') acc[p.type] = p.value;
        return acc;
      }, /** @type {Record<string,string>} */ ({}));
      return `${parts.day}${sep}${parts.month}${sep}${parts.year}`;
    }

    const url = new URL(req.url);

    const origin = req.headers.get('Origin') || '';
    const allow = new Set([
      'http://localhost:5173',
      'http://127.0.0.1:5173',
      'http://localhost:8000',
      'https://perceived-time-experiment.vercel.app',
      'https://sahilshindev.me',
      'https://www.sahilshindev.me',
      'https://assign.sahilshindev.me'
    ]);
    const allowOrigin = allow.has(origin) ? origin : 'https://perceived-time-experiment.vercel.app';
    const cors = {
      'Access-Control-Allow-Origin': allowOrigin,
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Credentials': 'true',
      'Access-Control-Max-Age': '600',
      'Vary': 'Origin'
    };
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }

    // Issue or return a server-generated participant ID; sets a cookie for convenience
    if (req.method === 'GET' && url.pathname === '/session') {
      const cookieStr = req.headers.get('Cookie') || '';
      const cookies = Object.fromEntries(
        cookieStr.split(/;\s*/).filter(Boolean).map(c => {
          const i = c.indexOf('=');
          const k = decodeURIComponent(i >= 0 ? c.slice(0, i) : c);
          const v = decodeURIComponent(i >= 0 ? c.slice(i + 1) : '');
          return [k, v];
        })
      );
      let pid = cookies['pt_pid'];
      if (!pid) {
        const uuid = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`).replace(/-/g, '');
        pid = `P${uuid.slice(0, 12)}`;
      }
      return new Response(JSON.stringify({ participantId: pid }), {
        headers: {
          ...cors,
          'Content-Type': 'application/json',
          // Cross-site cookie; fetch must use credentials: 'include' from the frontend
          'Set-Cookie': `pt_pid=${encodeURIComponent(pid)}; Path=/; Max-Age=31536000; SameSite=None; Secure`
        }
      });
    }

    // Check authoritative submission state for a participant
    if (req.method === 'GET' && url.pathname === '/submit-status') {
      const participantId = (url.searchParams.get('participantId') || '').trim();
      if (!participantId) {
        return new Response(JSON.stringify({ error: 'participantId required' }), {
          status: 400, headers: { 'Content-Type': 'application/json', ...cors }
        });
      }

      const row = await env.DB
        .prepare('SELECT payload FROM assignments WHERE participant_id = ?')
        .bind(participantId)
        .first();

      let payload = {};
      try { payload = row?.payload ? JSON.parse(row.payload) : {}; } catch {}

      return new Response(JSON.stringify({
        participantId,
        submitted: payload?.submitted === true,
        submittedAt: payload?.submittedAt || null,
        lastKey: payload?.lastKey || null
      }), {
        status: 200, headers: { 'Content-Type': 'application/json', ...cors }
      });
    }

    // Secure download route for combined reports: GET /download/:token
    if (req.method === 'GET' && url.pathname.startsWith('/download/')) {
      const token = url.pathname.replace('/download/', '').trim();
      if (!token) return new Response('Not Found', { status: 404, headers: cors });
      const row = await env.DB.prepare('SELECT k, expires_at FROM download_tokens WHERE token = ?')
        .bind(token)
        .first();
      if (!row) return new Response('Not Found', { status: 404, headers: cors });
      if (typeof row.expires_at === 'number' ? row.expires_at < Date.now() : Number(row.expires_at) < Date.now()) {
        return new Response('Link expired', { status: 410, headers: cors });
      }
      const key = row.k;
      const obj = await env.SUBMISSIONS.get(key);
      if (!obj) return new Response('Not Found', { status: 404, headers: cors });
      const filename = key.split('/').pop() || 'report.csv';
      const headers = {
        ...cors,
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': `attachment; filename="${filename}"`
      };
      // R2 get() returns an R2Object or R2ObjectBody; if body not present, .body is undefined
      // In Workers, Response can take the R2 body stream directly
      // @ts-ignore - body exists when requesting object body
      return new Response(obj.body || obj, { status: 200, headers });
    }

    if (url.pathname === '/submit' && req.method === 'POST') {
      const body = await req.json().catch(() => null);
      if (!body?.csv || !body?.participantId) {
        return new Response(JSON.stringify({ error: 'csv and participantId required' }), {
          status: 400, headers: { 'Content-Type': 'application/json', ...cors }
        });
      }

      // Load participant assignments first to validate state/token before writing to R2
      const row = await env.DB
        .prepare('SELECT payload FROM assignments WHERE participant_id = ?')
        .bind(body.participantId)
        .first();

      let payload = {};
      try { payload = row?.payload ? JSON.parse(row.payload) : {}; } catch {}

      // If already submitted, do not accept another upload
      if (payload?.submitted === true) {
        return new Response(JSON.stringify({ ok: false, error: 'already_submitted', alreadySubmitted: true }), {
          status: 409, headers: { 'Content-Type': 'application/json', ...cors }
        });
      }

      // Verify submission token when present in stored payload (backward compatible if absent)
      if (payload?.submissionToken) {
        const providedToken = body.submissionToken || '';
        if (providedToken !== payload.submissionToken) {
          return new Response(JSON.stringify({ error: 'invalid_submission_token' }), {
            status: 403, headers: { 'Content-Type': 'application/json', ...cors }
          });
        }
      }

      const fname = (body.filename || `${body.participantId}-${Date.now()}.csv`).replace(/[^\w\-.]/g,'_');
      const day = dayInTZ(new Date(), TZ); // BST/GMT aware
      const key = `submissions/${day}/${fname}`;

      // Store CSV after validation
      await env.SUBMISSIONS.put(key, body.csv, {
        httpMetadata: { contentType: 'text/csv; charset=utf-8' }
      });

      const assigned = Array.isArray(payload?.assignments) ? payload.assignments : [];

      // Atomically increment counts for the assigned pairs and mark submitted:true; invalidate token
      const updates = assigned.map(a =>
        env.DB.prepare('UPDATE counts SET n = n + 1 WHERE clip_id = ? AND format = ?')
             .bind(String(a.clipId), a.format)
      );
      const newPayload = JSON.stringify({ ...payload, submitted: true, submittedAt: Date.now(), lastKey: key, submissionToken: null });
      updates.push(
        env.DB.prepare('UPDATE assignments SET payload = ? WHERE participant_id = ?')
              .bind(newPayload, body.participantId)
      );
      if (updates.length) {
        await env.DB.batch(updates);
      } else {
        // No assignments found; still record submission flag
        await env.DB.prepare('UPDATE assignments SET payload = ? WHERE participant_id = ?')
          .bind(newPayload, body.participantId)
          .run();
      }

      return new Response(JSON.stringify({ ok: true, key, incremented: assigned.length }), {
        status: 200, headers: { 'Content-Type': 'application/json', ...cors }
      });
    }

    if (req.method !== 'POST') return json({ error: 'Use POST' }, 405, cors);

    let body; try { body = await req.json(); } catch { return json({ error: 'Bad JSON' }, 400, cors); }
    const { participantId, clips, formats = ['AV','A','V'] } = body || {};
    if (!participantId || !Array.isArray(clips) || clips.length === 0) return json({ error: 'participantId and clips[] required' }, 400, cors);

    const existing = await env.DB.prepare('SELECT payload FROM assignments WHERE participant_id = ?').bind(participantId).first();
    if (existing?.payload) return new Response(existing.payload, { headers: { 'Content-Type': 'application/json', ...cors } });

    await env.DB.batch(clips.flatMap(id => formats.map(f => env.DB.prepare('INSERT OR IGNORE INTO counts (clip_id, format, n) VALUES (?, ?, 0)').bind(id, f))));
    const assignments = [];
    for (const clipId of clips) {
      const rows = await env.DB.prepare('SELECT format, n FROM counts WHERE clip_id = ?').bind(clipId).all();
      const min = Math.min(...rows.results.map(r => r.n));
      const cands = rows.results.filter(r => r.n === min).map(r => r.format);
      const idx = detPick(`${participantId}:${clipId}`, cands.length);
      assignments.push({ clipId, format: cands[idx] });
    }
    // Save assignments with submitted:false and a submission token (no counts increment yet)
    const submissionToken = (crypto && crypto.randomUUID) ? crypto.randomUUID() : makeToken(36);
    await env.DB
      .prepare('INSERT INTO assignments (participant_id, payload) VALUES (?, ?)')
      .bind(participantId, JSON.stringify({ assignments, submitted: false, submissionToken, createdAt: Date.now() }))
      .run();
    return json({ assignments, submitted: false, submissionToken }, 200, cors);
  },

  // Nightly cron job to combine previous day's CSVs and email a download link via Resend
  async scheduled(event, env, ctx) {
    const TZ = env.REPORT_TZ || 'Europe/London';
    function dayInTZ(d, tz = 'Europe/London', sep = '-') {
      const parts = new Intl.DateTimeFormat('en-GB', {
        timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit'
      }).formatToParts(d).reduce((acc, p) => { if (p.type !== 'literal') acc[p.type] = p.value; return acc; }, /** @type {Record<string,string>} */ ({}));
      return `${parts.day}${sep}${parts.month}${sep}${parts.year}`;
    }

    // Use "yesterday" in the reporting TZ
    const now = new Date();
    const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const day = dayInTZ(yesterday, TZ);
    const prefix = `submissions/${day}/`;

    // List objects for the day
    const listed = await env.SUBMISSIONS.list({ prefix });
    if (!listed.objects || listed.objects.length === 0) {
      // nothing to combine; exit quietly
      return;
    }

    // Combine CSVs, keeping the header from the first file only
    let combined = '';
    let haveHeader = false;
    for (const obj of listed.objects) {
      const body = await env.SUBMISSIONS.get(obj.key);
      if (!body) continue;
      // @ts-ignore - body text() exists on R2ObjectBody
      const csv = await body.text();
      if (!csv) continue;
      const nl = csv.indexOf('\n');
      if (!haveHeader) {
        combined += csv;
        haveHeader = true;
      } else {
        const rest = nl >= 0 ? csv.slice(nl + 1) : '';
        if (rest.trim()) combined += (combined.endsWith('\n') ? '' : '\n') + rest;
      }
    }

    if (!combined) return;

    const reportKey = `reports/${day}/combined.csv`;
    await env.SUBMISSIONS.put(reportKey, combined, {
      httpMetadata: { contentType: 'text/csv; charset=utf-8' }
    });

    // Create a download token valid for 7 days
    const token = makeToken();
    const expiresAt = Date.now() + 7 * 24 * 60 * 60 * 1000;
    try {
      await env.DB.prepare('INSERT OR REPLACE INTO download_tokens (token, k, expires_at) VALUES (?, ?, ?)')
        .bind(token, reportKey, expiresAt)
        .run();
    } catch (e) {
      // Table may not exist yet; skip token + email if so
      console.error('download_tokens insert failed:', e);
      return;
    }

    // Email the link via Resend if configured
    const base = env.PUBLIC_BASE_URL; // e.g., https://assign.sahilshindev.me
    const to = env.REPORT_TO;
    const from = env.REPORT_FROM;
    const apiKey = env.RESEND_API_KEY;
    if (base && to && from && apiKey) {
      const url = `${base.replace(/\/$/, '')}/download/${token}`;
      const subject = `Perceived-time daily report for ${day}`;
      const text = `Your combined CSV report for ${day} is ready.\n\nDownload: ${url}\n\nThis link expires in 7 days.`;
      const res = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ from, to, subject, text })
      });
      if (!res.ok) {
        console.error('Resend email failed', res.status, await res.text());
      }
    }
  }
};
function json(obj, status, headers){ return new Response(JSON.stringify(obj), { status, headers: { 'Content-Type':'application/json', ...headers } }); }
function detPick(seed, n){ let h=2166136261; for(let i=0;i<seed.length;i++){ h^=seed.charCodeAt(i); h=Math.imul(h,16777619);} return Math.abs(h)%n; }

// Generate a URL-safe random token
function makeToken(len = 32) {
  const bytes = new Uint8Array(len);
  crypto.getRandomValues(bytes);
  const tryToChar = (b) => {
    const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_';
    return chars[b % chars.length];
  };
  return Array.from(bytes, tryToChar).join('');
}