/* Underpainting — value and shape studies from a reference photo.
   Everything runs locally; no image ever leaves the browser. */
(function () {
  'use strict';

  /* ============================================================
     WORKER
     Written as a real function rather than a string so it stays
     readable and lintable, then stringified into a Blob below.
     It must not reference anything outside itself.
     ============================================================ */
  function workerBody() {
    'use strict';

    /* ---------- sRGB -> CIELAB (D65) ---------- */
    var LIN = new Float32Array(256);
    for (var q = 0; q < 256; q++) {
      var cv = q / 255;
      LIN[q] = cv <= 0.04045 ? cv / 12.92 : Math.pow((cv + 0.055) / 1.055, 2.4);
    }
    function fLab(t) {
      return t > 0.008856451679035631 ? Math.cbrt(t) : t * 7.787037037037036 + 0.13793103448275862;
    }
    function toLab(rgba, n, L, A, B) {
      for (var i = 0; i < n; i++) {
        var o = i * 4;
        var r = LIN[rgba[o]], g = LIN[rgba[o + 1]], b = LIN[rgba[o + 2]];
        var X = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
        var Y =  r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        var Z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883;
        var fx = fLab(X), fy = fLab(Y), fz = fLab(Z);
        L[i] = 116 * fy - 16;
        A[i] = 500 * (fx - fy);
        B[i] = 200 * (fy - fz);
      }
    }

    /* ---------- separable box blur (3 passes ~ gaussian) ---------- */
    function boxBlurF(src, w, h, r) {
      if (r < 1) return src;
      var n = w * h, tmp = new Float32Array(n), out = new Float32Array(n), i, x, y, row, sum, cnt, add, sub;
      for (y = 0; y < h; y++) {
        row = y * w; sum = 0; cnt = 0;
        for (x = 0; x <= r && x < w; x++) { sum += src[row + x]; cnt++; }
        tmp[row] = sum / cnt;
        for (x = 1; x < w; x++) {
          sub = x - r - 1; if (sub >= 0) { sum -= src[row + sub]; cnt--; }
          add = x + r;     if (add < w)  { sum += src[row + add]; cnt++; }
          tmp[row + x] = sum / cnt;
        }
      }
      for (x = 0; x < w; x++) {
        sum = 0; cnt = 0;
        for (y = 0; y <= r && y < h; y++) { sum += tmp[y * w + x]; cnt++; }
        out[x] = sum / cnt;
        for (y = 1; y < h; y++) {
          sub = y - r - 1; if (sub >= 0) { sum -= tmp[sub * w + x]; cnt--; }
          add = y + r;     if (add < h)  { sum += tmp[add * w + x]; cnt++; }
          out[y * w + x] = sum / cnt;
        }
      }
      return out;
    }

    function blurRGBA(rgba, w, h, r) {
      if (r < 1) return rgba;
      var n = w * h, ch = [new Float32Array(n), new Float32Array(n), new Float32Array(n)], c, i;
      for (i = 0; i < n; i++) {
        ch[0][i] = rgba[i * 4]; ch[1][i] = rgba[i * 4 + 1]; ch[2][i] = rgba[i * 4 + 2];
      }
      var per = Math.max(1, Math.round(r / 3));
      for (c = 0; c < 3; c++) {
        var v = ch[c];
        for (var p = 0; p < 3; p++) v = boxBlurF(v, w, h, per);
        ch[c] = v;
      }
      var out = new Uint8ClampedArray(rgba.length);
      for (i = 0; i < n; i++) {
        var o = i * 4;
        out[o] = ch[0][i]; out[o + 1] = ch[1][i]; out[o + 2] = ch[2][i]; out[o + 3] = rgba[o + 3];
      }
      return out;
    }

    /* ---------- 3-value study ---------- */
    /* L* = 50 lands on sRGB 119, the true perceptual midpoint. */
    var TONES = [0, 119, 255];
    function valueStudy(rgba, w, h, blurRadius) {
      var src = blurRGBA(rgba, w, h, blurRadius);
      var n = w * h, L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
      toLab(src, n, L, A, B);
      var out = new Uint8ClampedArray(n * 4);
      for (var i = 0; i < n; i++) {
        var t = L[i] < 33.3333 ? TONES[0] : (L[i] < 66.6667 ? TONES[1] : TONES[2]);
        var o = i * 4;
        out[o] = t; out[o + 1] = t; out[o + 2] = t; out[o + 3] = rgba[o + 3];
      }
      return out;
    }

    /* ---------- saliency: where the detail lives ---------- */
    function saliency(L, A, B, w, h) {
      var n = w * h, i, mL = 0, mA = 0, mB = 0;
      for (i = 0; i < n; i++) { mL += L[i]; mA += A[i]; mB += B[i]; }
      mL /= n; mA /= n; mB /= n;

      var glob = new Float32Array(n), gMax = 0;
      for (i = 0; i < n; i++) {
        var dl = L[i] - mL, da = A[i] - mA, db = B[i] - mB;
        var d = Math.sqrt(dl * dl + da * da + db * db);
        glob[i] = d; if (d > gMax) gMax = d;
      }
      if (gMax > 0) for (i = 0; i < n; i++) glob[i] /= gMax;

      var lr = Math.min(30, Math.max(5, Math.round(Math.min(w, h) * 0.05)));
      var bL = boxBlurF(L, w, h, lr), bA = boxBlurF(A, w, h, lr), bB = boxBlurF(B, w, h, lr);
      var loc = new Float32Array(n), lMax = 0;
      for (i = 0; i < n; i++) {
        var el2 = L[i] - bL[i], ea = A[i] - bA[i], eb = B[i] - bB[i];
        var e = Math.sqrt(el2 * el2 + ea * ea + eb * eb);
        loc[i] = e; if (e > lMax) lMax = e;
      }
      if (lMax > 0) for (i = 0; i < n; i++) loc[i] /= lMax;

      var cx = w / 2, cy = h / 2, sx = 2 * (w * 0.4) * (w * 0.4), sy = 2 * (h * 0.4) * (h * 0.4);
      var out = new Float32Array(n), lo = Infinity, hi = -Infinity;
      for (var y = 0; y < h; y++) {
        var dy = y - cy, yt = dy * dy / sy, row = y * w;
        for (var x = 0; x < w; x++) {
          var dx = x - cx;
          var ctr = Math.exp(-(dx * dx / sx + yt));
          var v = 0.4 * glob[row + x] + 0.4 * loc[row + x] + 0.2 * ctr;
          out[row + x] = v;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      if (hi > lo) for (i = 0; i < n; i++) out[i] = (out[i] - lo) / (hi - lo);
      return out;
    }

    /* ---------- connected components over a label map ---------- */
    function components(labels, w, h) {
      var n = w * h, comp = new Int32Array(n).fill(-1), sizes = [], stack = new Int32Array(n), nc = 0;
      for (var s = 0; s < n; s++) {
        if (comp[s] !== -1) continue;
        var lab = labels[s], sp = 0, size = 0;
        stack[sp++] = s; comp[s] = nc;
        while (sp > 0) {
          var p = stack[--sp]; size++;
          var px = p % w, py = (p / w) | 0;
          if (px > 0     && comp[p - 1] === -1 && labels[p - 1] === lab) { comp[p - 1] = nc; stack[sp++] = p - 1; }
          if (px < w - 1 && comp[p + 1] === -1 && labels[p + 1] === lab) { comp[p + 1] = nc; stack[sp++] = p + 1; }
          if (py > 0     && comp[p - w] === -1 && labels[p - w] === lab) { comp[p - w] = nc; stack[sp++] = p - w; }
          if (py < h - 1 && comp[p + w] === -1 && labels[p + w] === lab) { comp[p + w] = nc; stack[sp++] = p + w; }
        }
        sizes.push(size); nc++;
      }
      return { comp: comp, sizes: sizes, count: nc };
    }

    /* ---------- absorb regions below the minimum size ---------- */
    function enforceMinSize(labels, w, h, minSize, passes) {
      var n = w * h;
      for (var pass = 0; pass < passes; pass++) {
        var cc = components(labels, w, h);
        var comp = cc.comp, sizes = cc.sizes, nc = cc.count;
        if (nc <= 1) return;

        var floors = new Float32Array(nc).fill(Infinity), i, c;
        for (i = 0; i < n; i++) { c = comp[i]; if (minSize[i] < floors[c]) floors[c] = minSize[i]; }

        var small = [];
        for (c = 0; c < nc; c++) if (sizes[c] < floors[c]) small.push(c);
        if (!small.length) return;

        var isSmall = new Uint8Array(nc);
        for (i = 0; i < small.length; i++) isSmall[small[i]] = 1;

        /* tally shared border between each small component and its neighbours */
        var tally = new Map();
        for (var y = 0; y < h; y++) {
          for (var x = 0; x < w; x++) {
            var p = y * w + x, a = comp[p];
            if (!isSmall[a]) continue;
            for (var k = 0; k < 4; k++) {
              var nx = x + (k === 0 ? -1 : k === 1 ? 1 : 0);
              var ny = y + (k === 2 ? -1 : k === 3 ? 1 : 0);
              if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
              var b = comp[ny * w + nx];
              if (b === a) continue;
              var m = tally.get(a); if (!m) { m = new Map(); tally.set(a, m); }
              m.set(b, (m.get(b) || 0) + 1);
            }
          }
        }

        /* each small component adopts the label of its longest neighbour,
           preferring neighbours that are not themselves being absorbed */
        var adopt = new Int32Array(nc).fill(-1), changed = false;
        tally.forEach(function (m2, a2) {
          var bestBig = -1, bestBigN = -1, bestAny = -1, bestAnyN = -1;
          m2.forEach(function (count, b2) {
            if (count > bestAnyN) { bestAnyN = count; bestAny = b2; }
            if (!isSmall[b2] && count > bestBigN) { bestBigN = count; bestBig = b2; }
          });
          var target = bestBig >= 0 ? bestBig : bestAny;
          if (target >= 0 && sizes[target] >= sizes[a2]) { adopt[a2] = target; changed = true; }
        });
        if (!changed) return;

        var repLabel = new Int32Array(nc).fill(-1);
        for (i = 0; i < n; i++) if (repLabel[comp[i]] === -1) repLabel[comp[i]] = labels[i];
        for (i = 0; i < n; i++) { var t = adopt[comp[i]]; if (t >= 0) labels[i] = repLabel[t]; }
      }
    }

    /* ---------- SLIC with saliency-adaptive seeding ---------- */
    function slic(rgba, w, h, k, opts) {
      opts = opts || {};
      var compactness = opts.compactness || 10;
      var maxIter = opts.iterations || 7;
      /* Shapes are found on `rgba` (possibly smoothed); colours are sampled
         from `trueColour` (never smoothed) so region colours stay accurate. */
      var trueColour = opts.trueColour || rgba;
      var n = w * h;
      var S = Math.round(Math.sqrt(n / Math.max(1, k)));
      if (S < 2) return rgba.slice();

      var L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
      toLab(rgba, n, L, A, B);
      post('Reading colour');

      /* gradient of lightness, for nudging seeds off edges */
      var grad = new Float32Array(n), x, y, i;
      for (y = 1; y < h - 1; y++) {
        for (x = 1; x < w - 1; x++) {
          i = y * w + x;
          var gx = L[i + 1] - L[i - 1], gy = L[i + w] - L[i - w];
          grad[i] = gx * gx + gy * gy;
        }
      }

      /* where is the detail? busy areas earn extra seeds */
      var sal = saliency(L, A, B, w, h);
      var sr = Math.min(50, Math.max(5, Math.round(Math.min(w, h) * 0.05)));
      sal = boxBlurF(sal, w, h, sr);
      var lo = Infinity, hi = -Infinity;
      for (i = 0; i < n; i++) { if (sal[i] < lo) lo = sal[i]; if (sal[i] > hi) hi = sal[i]; }
      if (hi > lo) for (i = 0; i < n; i++) sal[i] = (sal[i] - lo) / (hi - lo);
      var busy = new Uint8Array(n);
      for (i = 0; i < n; i++) if (sal[i] > 0.6) busy[i] = 1;
      post('Finding detail');

      var cxs = [], cys = [], cL = [], cA = [], cB = [];
      function seed(sx, sy) {
        var bestG = Infinity, bx = sx, by = sy;
        for (var oy = -1; oy <= 1; oy++) {
          var yy = sy + oy; if (yy < 1 || yy >= h - 1) continue;
          for (var ox = -1; ox <= 1; ox++) {
            var xx = sx + ox; if (xx < 1 || xx >= w - 1) continue;
            var g = grad[yy * w + xx];
            if (g < bestG) { bestG = g; bx = xx; by = yy; }
          }
        }
        var p = by * w + bx;
        cxs.push(bx); cys.push(by); cL.push(L[p]); cA.push(A[p]); cB.push(B[p]);
      }

      var half = S >> 1;
      for (y = half; y < h; y += S) for (x = half; x < w; x += S) seed(x, y);

      /* extra seeds inside busy regions, packed tighter */
      var fine = Math.min(S - 1, Math.max(3, Math.round(S / Math.SQRT2)));
      if (fine >= 3 && fine < S) {
        var bc = components(busy, w, h);
        var keep = new Uint8Array(bc.count);
        for (i = 0; i < bc.count; i++) if (bc.sizes[i] >= 500) keep[i] = 1;
        var fh = fine >> 1;
        for (y = fh; y < h; y += fine) {
          for (x = fh; x < w; x += fine) {
            i = y * w + x;
            if (busy[i] && keep[bc.comp[i]]) seed(x, y);
          }
        }
      }

      var K = cxs.length;
      if (K === 0) return rgba.slice();
      var fx = Float32Array.from(cxs), fy = Float32Array.from(cys);
      var fL = Float32Array.from(cL), fA = Float32Array.from(cA), fB = Float32Array.from(cB);
      post('Placing ' + K + ' seeds');

      var labels = new Int32Array(n).fill(-1);
      var dist = new Float32Array(n);
      var prev = new Int32Array(n);
      var sw = (compactness / S) * (compactness / S);
      var cnt = new Int32Array(K), sx2 = new Float64Array(K), sy2 = new Float64Array(K);
      var sl = new Float64Array(K), sa = new Float64Array(K), sb = new Float64Array(K);

      for (var iter = 0; iter < maxIter; iter++) {
        prev.set(labels);
        dist.fill(Infinity);

        for (var c = 0; c < K; c++) {
          var ccx = fx[c], ccy = fy[c], clL = fL[c], clA = fA[c], clB = fB[c];
          var x0 = Math.max(0, Math.floor(ccx - S)), x1 = Math.min(w - 1, Math.ceil(ccx + S));
          var y0 = Math.max(0, Math.floor(ccy - S)), y1 = Math.min(h - 1, Math.ceil(ccy + S));
          for (y = y0; y <= y1; y++) {
            var row = y * w, dyy = y - ccy, dy2 = dyy * dyy;
            for (x = x0; x <= x1; x++) {
              i = row + x;
              var dl = L[i] - clL, da = A[i] - clA, db = B[i] - clB, dxx = x - ccx;
              var d = dl * dl + da * da + db * db + (dxx * dxx + dy2) * sw;
              if (d < dist[i]) { dist[i] = d; labels[i] = c; }
            }
          }
        }

        cnt.fill(0); sx2.fill(0); sy2.fill(0); sl.fill(0); sa.fill(0); sb.fill(0);
        var moved = 0;
        for (y = 0; y < h; y++) {
          for (x = 0; x < w; x++) {
            i = y * w + x; var lb = labels[i];
            if (lb < 0) continue;
            cnt[lb]++; sx2[lb] += x; sy2[lb] += y; sl[lb] += L[i]; sa[lb] += A[i]; sb[lb] += B[i];
          }
        }
        for (i = 0; i < n; i++) if (labels[i] !== prev[i]) moved++;
        for (c = 0; c < K; c++) {
          var m3 = cnt[c]; if (!m3) continue;
          fx[c] = sx2[c] / m3; fy[c] = sy2[c] / m3;
          fL[c] = sl[c] / m3;  fA[c] = sa[c] / m3;  fB[c] = sb[c] / m3;
        }

        post('Refining shapes');
        if (iter >= 2 && (moved * 100 / n) < 5) break;
      }

      /* stragglers adopt a neighbour */
      var OX = [-1, 1, 0, 0, -1, 1, -1, 1], OY = [0, 0, -1, 1, -1, -1, 1, 1];
      for (var guard = 0; guard < 50; guard++) {
        var left = false;
        for (i = 0; i < n; i++) {
          if (labels[i] >= 0) continue;
          var px = i % w, py = (i / w) | 0, got = false;
          for (var qq = 0; qq < 8; qq++) {
            var nx2 = px + OX[qq], ny2 = py + OY[qq];
            if (nx2 < 0 || nx2 >= w || ny2 < 0 || ny2 >= h) continue;
            var nl = labels[ny2 * w + nx2];
            if (nl >= 0) { labels[i] = nl; got = true; break; }
          }
          if (!got) left = true;
        }
        if (!left) break;
      }

      /* clean up specks: busy areas are allowed much smaller shapes */
      var avg = n / K, base = avg * 0.5, tight = base / 8;
      var minSize = new Float32Array(n);
      for (i = 0; i < n; i++) minSize[i] = busy[i] ? tight : base;
      enforceMinSize(labels, w, h, minSize, 10);
      post('Cleaning up');

      /* flat-fill each region with the mean of the ORIGINAL colours under it */
      var cc2 = components(labels, w, h);
      var nc = cc2.count, comp = cc2.comp;
      var rs = new Float64Array(nc), gs = new Float64Array(nc), bs = new Float64Array(nc), ns = new Int32Array(nc);
      for (i = 0; i < n; i++) {
        var cm = comp[i], o = i * 4;
        rs[cm] += trueColour[o]; gs[cm] += trueColour[o + 1]; bs[cm] += trueColour[o + 2]; ns[cm]++;
      }
      var out = new Uint8ClampedArray(n * 4);
      for (i = 0; i < n; i++) {
        var cm2 = comp[i], o2 = i * 4, m4 = ns[cm2] || 1;
        out[o2] = rs[cm2] / m4; out[o2 + 1] = gs[cm2] / m4; out[o2 + 2] = bs[cm2] / m4;
        out[o2 + 3] = trueColour[o2 + 3];
      }
      return out;
    }

    function post(text) { self.postMessage({ type: 'progress', text: text }); }

    self.onmessage = function (e) {
      var d = e.data, out;
      try {
        if (d.mode === 'values') {
          post('Sorting values');
          out = valueStudy(d.rgba, d.w, d.h, d.blur);
        } else {
          var src = d.blur > 0 ? blurRGBA(d.rgba, d.w, d.h, d.blur) : d.rgba;
          out = slic(src, d.w, d.h, d.segments, {
            compactness: 10, iterations: 7, trueColour: d.rgba
          });
        }
        self.postMessage({ type: 'done', gen: d.gen, slot: d.slot, rgba: out, w: d.w, h: d.h }, [out.buffer]);
      } catch (err) {
        self.postMessage({ type: 'error', gen: d.gen, slot: d.slot, message: String(err && err.message || err) });
      }
    };
  }

  var WORKER_SRC = '(' + workerBody.toString() + ')();';


  /* ============================================================
     APP
     ============================================================ */

  var MAX_DIM = 1400;

  var el = {
    stage: document.getElementById('stage'),
    pair: document.getElementById('pair'),
    viewO: document.getElementById('viewO'),
    viewA: document.getElementById('viewA'),
    viewB: document.getElementById('viewB'),
    drop: document.getElementById('drop'),
    busy: document.getElementById('busy'),
    busyText: document.getElementById('busy-text'),
    file: document.getElementById('file'),
    pick: document.getElementById('pick'),
    replace: document.getElementById('replace'),
    save: document.getElementById('save'),
    detail: document.getElementById('detail'),
    detailOut: document.getElementById('detail-out')
  };

  var ctxO = el.viewO.getContext('2d', { willReadFrequently: true });
  var ctxA = el.viewA.getContext('2d', { willReadFrequently: true });
  var ctxB = el.viewB.getContext('2d', { willReadFrequently: true });

  var state = {
    source: null,       // ImageData of the scaled original
    w: 0, h: 0,
    blur: 0,            // shared by both panes
    detail: 10,
    gen: 0,             // bumped per render; stale results are dropped
    pending: 0,
    renderedA: null,    // values pane
    renderedB: null     // simplified pane
  };

  /* Details 1-14 -> segment count, geometric from 8 to 3500.
     Levels 1-4 sit below 45, where SLIC has no colour cue to break ties in
     smooth areas, so shapes there follow the seed grid more than the image. */
  function segmentsFor(level) {
    var t = (level - 1) / 13;
    return Math.round(8 * Math.pow(3500 / 8, t));
  }

  function cappedSegments(level) {
    var cap = state.w ? Math.max(1, Math.floor(state.w * state.h / 16)) : Infinity;
    return Math.min(segmentsFor(level), cap);
  }

  function updateDetailOut() {
    el.detailOut.textContent = cappedSegments(state.detail).toLocaleString() + ' shapes';
  }

  function setBusy(on) { el.busy.classList.toggle('hidden', !on); }

  /* ---------- worker ---------- */
  var worker = null;
  function getWorker() {
    if (worker) return worker;
    var url = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'application/javascript' }));
    worker = new Worker(url);
    worker.onmessage = function (e) {
      var d = e.data;
      if (d.type === 'progress') { el.busyText.textContent = d.text; return; }
      if (d.gen !== state.gen) return;             // a newer render superseded this one
      if (d.type === 'error') {
        state.pending = 0;
        el.busyText.textContent = 'That image could not be processed. Try a smaller file.';
        setBusy(true);
        return;
      }

      var img = new ImageData(new Uint8ClampedArray(d.rgba), d.w, d.h);
      if (d.slot === 'A') { state.renderedA = img; ctxA.putImageData(img, 0, 0); }
      else                { state.renderedB = img; ctxB.putImageData(img, 0, 0); }

      if (--state.pending <= 0) {
        setBusy(false);
        el.save.disabled = false;
      }
    };
    return worker;
  }

  /* ---------- render ---------- */
  var timer = null;
  function render(immediate, only) {
    if (!state.source) return;
    clearTimeout(timer);
    var go = function () {
      var gen = ++state.gen;
      var jobs = [];
      /* the detail slider only affects the simplified pane, so leave the
         value study alone unless it has never been drawn */
      if (only !== 'B' || !state.renderedA) {
        jobs.push({ mode: 'values', slot: 'A', level: 1 });
      }
      jobs.push({ mode: 'simplify', slot: 'B', level: state.detail });

      state.pending = jobs.length;
      setBusy(true);
      el.busyText.textContent = 'Working';

      jobs.forEach(function (j) {
        var copy = new Uint8ClampedArray(state.source.data);
        getWorker().postMessage({
          gen: gen,
          slot: j.slot,
          mode: j.mode,
          rgba: copy,
          w: state.w,
          h: state.h,
          blur: state.blur,
          segments: cappedSegments(j.level)
        }, [copy.buffer]);
      });
    };
    if (immediate) go(); else timer = setTimeout(go, 180);
  }

  /* ---------- layout ---------- */
  function layoutPair() {
    if (!state.source) return;
    var r = el.stage.getBoundingClientRect();

    /* Phone: one pane per screenful, so each image gets the whole stage
       height and you swipe between them. The flex `height:100%` trick
       doesn't extend the scroll area, so set the height explicitly. */
    if (window.matchMedia('(max-width:820px)').matches) {
      var cs = getComputedStyle(el.stage);
      var inner = el.stage.clientHeight
                - (parseFloat(cs.paddingTop) || 0)
                - (parseFloat(cs.paddingBottom) || 0);
      if (inner > 0) el.pair.style.setProperty('--pane-h', Math.round(inner) + 'px');
      el.pair.classList.remove('stack');
      return;
    }

    /* Desktop: side by side wastes height on landscape images and stacking
       wastes width on portrait ones. Measure both and keep whichever draws bigger. */
    el.pair.style.removeProperty('--pane-h');
    var availW = r.width - 48, availH = r.height - 48;   // stage padding
    if (availW <= 0 || availH <= 0) return;
    var gap = 16, capH = 26, iw = state.w, ih = state.h, N = 3;
    var sideBySide = Math.min(((availW - gap * (N - 1)) / N) / iw, (availH - capH) / ih);
    var stacked    = Math.min(availW / iw, ((availH - gap * (N - 1) - N * capH) / N) / ih);
    el.pair.classList.toggle('stack', stacked > sideBySide);
  }
  window.addEventListener('resize', layoutPair);
  window.addEventListener('orientationchange', layoutPair);

  /* ---------- loading an image ---------- */
  function loadFile(file) {
    if (!file || !/^image\//.test(file.type)) return;
    var img = new Image();
    img.onload = function () {
      var scale = Math.min(1, MAX_DIM / Math.max(img.width, img.height));
      var w = Math.max(1, Math.round(img.width * scale));
      var h = Math.max(1, Math.round(img.height * scale));

      el.viewO.width = w; el.viewO.height = h;
      el.viewA.width = w; el.viewA.height = h;
      el.viewB.width = w; el.viewB.height = h;
      ctxO.drawImage(img, 0, 0, w, h);
      state.source = ctxO.getImageData(0, 0, w, h);
      state.w = w; state.h = h;
      state.renderedA = state.renderedB = null;

      el.drop.classList.add('hidden');
      el.pair.classList.remove('hidden');
      el.replace.disabled = false;
      el.stage.scrollTop = 0;        // a new image starts at the first pane
      layoutPair();
      updateDetailOut();
      render(true);
      URL.revokeObjectURL(img.src);
    };
    img.onerror = function () {
      el.busyText.textContent = 'That file could not be opened as an image';
      setBusy(true);
    };
    img.src = URL.createObjectURL(file);
  }

  el.pick.addEventListener('click', function () { el.file.click(); });
  el.replace.addEventListener('click', function () { el.file.click(); });
  el.file.addEventListener('change', function (e) { loadFile(e.target.files[0]); el.file.value = ''; });

  ['dragenter', 'dragover'].forEach(function (ev) {
    el.stage.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    el.stage.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.remove('over'); });
  });
  el.stage.addEventListener('drop', function (e) {
    if (e.dataTransfer && e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });

  /* ---------- smoothing ---------- */
  var smoothGroup = document.getElementById('smooth');
  smoothGroup.addEventListener('click', function (e) {
    var btn = e.target.closest('button');
    if (!btn) return;
    Array.prototype.forEach.call(smoothGroup.querySelectorAll('button'), function (b) {
      b.setAttribute('aria-pressed', String(b === btn));
    });
    state.blur = parseInt(btn.dataset.r, 10);
    render(true);
  });

  /* ---------- detail ---------- */
  el.detail.addEventListener('input', function () {
    state.detail = parseInt(el.detail.value, 10);
    updateDetailOut();
    render(false, 'B');
  });

  /* ---------- save ---------- */
  el.save.addEventListener('click', function () {
    if (!state.renderedA || !state.renderedB) return;
    var gap = 16, step = state.w + gap;
    var canvas = document.createElement('canvas');
    canvas.width = state.w * 3 + gap * 2;
    canvas.height = state.h;
    var cc = canvas.getContext('2d');
    cc.fillStyle = '#2b2b2b';
    cc.fillRect(0, 0, canvas.width, canvas.height);
    cc.putImageData(state.source, 0, 0);
    cc.putImageData(state.renderedA, step, 0);
    cc.putImageData(state.renderedB, step * 2, 0);
    canvas.toBlob(function (blob) {
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'underpainting-study.png';
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    }, 'image/png');
  });

  updateDetailOut();
})();