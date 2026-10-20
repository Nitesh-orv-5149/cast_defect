// CAD-reference inspection: the geometry checks of research/cad_reference/cadref.py, in dependency-free JS.
// Works on typed arrays only (no DOM), so the same file runs in the page and in a Node test harness.
(function (root) {
"use strict";

// ---------------------------------------------------------------- CAD loading

// Triangle soup: Float32Array of 9 numbers per triangle (x0 y0 z0 x1 y1 z1 x2 y2 z2), millimetres
function parseSTL(buf) {
  const dv = new DataView(buf), n = buf.byteLength >= 84 ? dv.getUint32(80, true) : 0;
  if (n > 0 && buf.byteLength === 84 + n * 50) {
    const t = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) for (let k = 0; k < 9; k++) t[i * 9 + k] = dv.getFloat32(84 + i * 50 + 12 + k * 4, true);
    return t;
  }
  const txt = new TextDecoder().decode(buf), v = [], re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g;
  for (let m; (m = re.exec(txt));) v.push(+m[1], +m[2], +m[3]);
  if (!v.length || v.length % 9 || v.some(Number.isNaN)) throw new Error("not a valid STL file");
  return Float32Array.from(v);
}

function parseOBJ(text) {
  const vs = [], out = [];
  for (const line of text.split("\n")) {
    const p = line.trim().split(/\s+/);
    if (p[0] === "v") vs.push([+p[1], +p[2], +p[3]]);
    else if (p[0] === "f") {
      const idx = p.slice(1).map(s => { const i = parseInt(s, 10); return i < 0 ? vs.length + i : i - 1; });
      for (let k = 1; k + 1 < idx.length; k++) for (const i of [idx[0], idx[k], idx[k + 1]]) out.push(...vs[i]);
    }
  }
  if (!out.length) throw new Error("no faces in OBJ file");
  return Float32Array.from(out);
}

function loadCAD(name, buf) {
  return /\.obj$/i.test(name) ? parseOBJ(new TextDecoder().decode(buf)) : parseSTL(buf);
}

// Which face of the CAD the camera looks at: rotate that face's outward normal onto +Z
const FACES = {
  top: v => [v[0], v[1], v[2]], bottom: v => [v[0], -v[1], -v[2]],
  front: v => [v[0], v[2], -v[1]], back: v => [v[0], -v[2], v[1]],
  right: v => [-v[2], v[1], v[0]], left: v => [v[2], v[1], -v[0]],
};

// Orient to the chosen face, centre the footprint on the origin, rest the part on z = 0
function prepare(tris, face) {
  const f = FACES[face] || FACES.top, t = new Float32Array(tris.length);
  for (let i = 0; i < tris.length; i += 3) { const r = f([tris[i], tris[i + 1], tris[i + 2]]); t[i] = r[0]; t[i + 1] = r[1]; t[i + 2] = r[2]; }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < t.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], t[i + k]); hi[k] = Math.max(hi[k], t[i + k]); }
  const off = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]];
  for (let i = 0; i < t.length; i += 3) for (let k = 0; k < 3; k++) t[i + k] -= off[k];
  return { tris: t, size: [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]] };
}

// ---------------------------------------------------------------- grid helpers (w x h, row-major)

function blur(src, w, h, sigma, radius) {
  const r = radius ?? Math.max(1, Math.ceil(3 * sigma)), k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) s += (k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)));
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const cx = Int32Array.from({ length: w + 2 * r }, (_, i) => Math.min(w - 1, Math.max(0, i - r)));
  const cy = Int32Array.from({ length: h + 2 * r }, (_, i) => Math.min(h - 1, Math.max(0, i - r)));
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) { let a = 0; for (let i = 0; i <= 2 * r; i++) a += k[i] * src[row + cx[x + i]]; tmp[row + x] = a; }
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = 0; i <= 2 * r; i++) a += k[i] * tmp[cy[y + i] * w + x];
    out[y * w + x] = a;
  }
  return out;
}

function sobel(src, w, h) {
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  const at = (x, y) => src[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    gx[y * w + x] = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
    gy[y * w + x] = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
  }
  return [gx, gy];
}

// Separable square max/min filter; pixels outside the image are ignored.
// Binary masks (Uint8Array) use a sliding window count, O(1) per pixel; float maps use a direct loop.
function rank(src, w, h, k, isMax) {
  const r = k >> 1;
  if (src instanceof Uint8Array) {
    const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
    const pass = (inp, o, len, n, stride, step) => {
      for (let line = 0; line < n; line++) {
        const base = line * stride;
        let sum = 0, cnt = 0;
        for (let i = 0; i <= Math.min(len - 1, r); i++) { sum += inp[base + i * step] ? 1 : 0; cnt++; }
        for (let i = 0; i < len; i++) {
          o[base + i * step] = isMax ? (sum > 0 ? 1 : 0) : (sum === cnt ? 1 : 0);
          const add = i + r + 1, rem = i - r;
          if (add < len) { sum += inp[base + add * step] ? 1 : 0; cnt++; }
          if (rem >= 0) { sum -= inp[base + rem * step] ? 1 : 0; cnt--; }
        }
      }
    };
    pass(src, tmp, w, h, w, 1);
    pass(tmp, out, h, w, 1, w);
    return out;
  }
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = src[y * w + x];
    const a = Math.max(0, x - r), b = Math.min(w - 1, x + r);
    if (isMax) { for (let i = a; i <= b; i++) { const t = src[y * w + i]; if (t > v) v = t; } }
    else { for (let i = a; i <= b; i++) { const t = src[y * w + i]; if (t < v) v = t; } }
    tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = tmp[y * w + x];
    const a = Math.max(0, y - r), b = Math.min(h - 1, y + r);
    if (isMax) { for (let i = a; i <= b; i++) { const t = tmp[i * w + x]; if (t > v) v = t; } }
    else { for (let i = a; i <= b; i++) { const t = tmp[i * w + x]; if (t < v) v = t; } }
    out[y * w + x] = v;
  }
  return out;
}
const dilate = (m, w, h, k) => rank(m, w, h, k, true);
const erode = (m, w, h, k) => rank(m, w, h, k, false);
const open = (m, w, h, k) => dilate(erode(m, w, h, k), w, h, k);
const close = (m, w, h, k) => erode(dilate(m, w, h, k), w, h, k);
const and = (...ms) => ms[0].map((v, i) => ms.every(m => m[i]) ? 1 : 0);
const not = m => m.map(v => v ? 0 : 1);
const or = (...ms) => ms[0].map((v, i) => ms.some(m => m[i]) ? 1 : 0);
const count = m => { let s = 0; for (let i = 0; i < m.length; i++) s += m[i] ? 1 : 0; return s; };

// Connected components of a binary mask: labels (0 = background) and pixel counts per label
function components(m, w, h, conn8) {
  const lab = new Int32Array(w * h), area = [0], q = new Int32Array(w * h);
  let n = 0;
  for (let s = 0; s < w * h; s++) {
    if (!m[s] || lab[s]) continue;
    n++; area.push(0);
    let head = 0, tail = 0; q[tail++] = s; lab[s] = n;
    while (head < tail) {
      const p = q[head++], x = p % w, y = (p / w) | 0;
      area[n]++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if ((!dx && !dy) || (!conn8 && dx && dy)) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const qq = yy * w + xx;
        if (m[qq] && !lab[qq]) { lab[qq] = n; q[tail++] = qq; }
      }
    }
  }
  return { lab, area, n };
}

// Background pixels not connected (4-way) to the image border = through-holes
function enclosedBackground(mask, w, h) {
  const bg = not(mask), { lab, n } = components(bg, w, h, false), border = new Set();
  for (let x = 0; x < w; x++) { border.add(lab[x]); border.add(lab[(h - 1) * w + x]); }
  for (let y = 0; y < h; y++) { border.add(lab[y * w]); border.add(lab[y * w + w - 1]); }
  return lab.map(l => (l && !border.has(l)) ? 1 : 0);
}

function quantile(values, q) {
  if (!values.length) return 0;
  const a = Float32Array.from(values).sort();
  return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1) + 0.5))];
}
const pick = (arr, m) => { const out = []; for (let i = 0; i < m.length; i++) if (m[i]) out.push(arr[i]); return out; };

function otsu(u8) {
  const hist = new Float64Array(256);
  for (const v of u8) hist[v]++;
  let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let wB = 0, sB = 0, best = 0, thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = u8.length - wB; if (!wF) break;
    sB += t * hist[t];
    const d = sB / wB - (sum - sB) / wF, v = wB * wF * d * d;
    if (v > best) { best = v; thr = t; }
  }
  return thr;
}

// ---------------------------------------------------------------- CAD rendering (z-buffer)

function viewRotation(tiltDeg, azimDeg) {
  const a = azimDeg * Math.PI / 180, t = tiltDeg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), ct = Math.cos(t), st = Math.sin(t);
  const Rz = [[c, -s, 0], [s, c, 0], [0, 0, 1]], Rx = [[1, 0, 0], [0, ct, -st], [0, st, ct]];
  const mul = (A, B) => A.map((r, i) => B[0].map((_, j) => r.reduce((acc, _, k) => acc + A[i][k] * B[k][j], 0)));
  const RzT = Rz[0].map((_, i) => Rz.map(r => r[i]));
  return mul(mul(RzT, Rx), Rz);
}

function render(tris, R, ppm, size, camDist) {
  const nt = tris.length / 9, c = (size - 1) / 2, N = size * size;
  const px = new Float32Array(nt * 3), py = new Float32Array(nt * 3), pz = new Float32Array(nt * 3);
  for (let i = 0; i < nt * 3; i++) {
    const X = tris[i * 3], Y = tris[i * 3 + 1], Z = tris[i * 3 + 2];
    const vx = R[0][0] * X + R[0][1] * Y + R[0][2] * Z, vy = R[1][0] * X + R[1][1] * Y + R[1][2] * Z, vz = R[2][0] * X + R[2][1] * Y + R[2][2] * Z;
    const k = camDist ? camDist / Math.max(camDist - vz, 1) : 1;  // closer = bigger, exact at z = 0
    px[i] = c + vx * ppm * k; py[i] = c - vy * ppm * k; pz[i] = vz;
  }
  const depth = new Float32Array(N).fill(-Infinity), fid = new Int32Array(N).fill(-1);
  const nx = new Float32Array(nt), ny = new Float32Array(nt), nz = new Float32Array(nt);
  for (let i = 0; i < nt; i++) {
    const a = i * 3, b = a + 1, d = a + 2;
    // view-space normal from the rotated (unprojected) vertices
    const A = rot(R, tris, a), B = rot(R, tris, b), C = rot(R, tris, d);
    let ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2], wx = C[0] - A[0], wy = C[1] - A[1], wz = C[2] - A[2];
    let cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx, l = Math.hypot(cx, cy, cz) + 1e-12;
    nx[i] = cx / l; ny[i] = cy / l; nz[i] = cz / l;
    const x0 = px[a], x1 = px[b], x2 = px[d], y0 = py[a], y1 = py[b], y2 = py[d];
    const den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (Math.abs(den) < 1e-9 || (!camDist && nz[i] <= 1e-6)) continue;
    const xa = Math.max(0, Math.floor(Math.min(x0, x1, x2))), xb = Math.min(size - 1, Math.ceil(Math.max(x0, x1, x2)));
    const ya = Math.max(0, Math.floor(Math.min(y0, y1, y2))), yb = Math.min(size - 1, Math.ceil(Math.max(y0, y1, y2)));
    for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
      const w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / den, w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / den, w2 = 1 - w0 - w1;
      if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
      const z = w0 * pz[a] + w1 * pz[b] + w2 * pz[d], p = y * size + x;
      if (z > depth[p]) { depth[p] = z; fid[p] = i; }
    }
  }
  const mask = new Uint8Array(N), dfin = new Float32Array(N);
  for (let p = 0; p < N; p++) if (fid[p] >= 0) { mask[p] = 1; dfin[p] = depth[p]; }
  const holes = enclosedBackground(mask, size, size);
  // Hole mouths: each through-hole plus the steep walls connected to it. At a slightly different camera angle the
  // background can show anywhere in the mouth, so only openings outside every mouth are unexpected.
  const mouth = components(holes, size, size, false).lab;
  const queue = [];
  for (let p = 0; p < N; p++) if (mouth[p]) queue.push(p);
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi], x = p % size, y = (p / size) | 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy, q = yy * size + xx;
      if (xx < 0 || yy < 0 || xx >= size || yy >= size || mouth[q] || !mask[q] || Math.abs(nz[fid[q]]) > 0.4) continue;
      mouth[q] = mouth[p]; queue.push(q);
    }
  }
  const outline = and(mask, not(erode(mask, size, size, 3)));
  const edges = outline.slice(), cosv = Math.cos(25 * Math.PI / 180);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const p = y * size + x;
    if (!mask[p]) continue;
    for (const q of [x + 1 < size ? p + 1 : -1, y + 1 < size ? p + size : -1]) {
      if (q < 0 || !mask[q]) continue;
      const f = fid[p], g = fid[q];
      if (nx[f] * nx[g] + ny[f] * ny[g] + nz[f] * nz[g] < cosv || Math.abs(dfin[p] - dfin[q]) > 0.8) edges[p] = 1;
    }
  }
  return { mask, holes, mouth, edges, outline, depth: dfin, size };
}
function rot(R, t, i) {
  const X = t[i * 3], Y = t[i * 3 + 1], Z = t[i * 3 + 2];
  return [R[0][0] * X + R[0][1] * Y + R[0][2] * Z, R[1][0] * X + R[1][1] * Y + R[1][2] * Z, R[2][0] * X + R[2][1] * Y + R[2][2] * Z];
}

// Unit normal of the CAD's edges, from the blurred depth map (background = a deep floor)
function edgeNormals(rd) {
  const S = rd.size, N = S * S;
  let mn = Infinity;
  for (let p = 0; p < N; p++) if (rd.mask[p]) mn = Math.min(mn, rd.depth[p]);
  const d = new Float32Array(N);
  for (let p = 0; p < N; p++) d[p] = rd.mask[p] ? rd.depth[p] : mn - 6;
  const [gx, gy] = sobel(blur(d, S, S, 1.5), S, S);
  for (let p = 0; p < N; p++) { const l = Math.hypot(gx[p], gy[p]) + 1e-6; gx[p] /= l; gy[p] /= l; }
  return [gx, gy];
}

// Coarse-search copy of a template: centred and scaled into a C x C grid
function centroidRadius(m, w, h) {
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) { sx += x; sy += y; n++; }
  return [sy / n, sx / n, Math.sqrt(n / Math.PI), n];
}

async function buildProfile(tris, opts = {}) {
  const size = opts.size || 320, camDist = opts.camDist || 0, tilts = opts.tilts || [0, 6, 12, 18], azims = opts.azims || 8, C = 96;
  let r = 0;
  for (let i = 0; i < tris.length; i += 3) r = Math.max(r, Math.hypot(tris[i], tris[i + 1]));
  const ppm = size * 0.42 / r, views = [], list = [];
  for (const t of tilts) for (const a of (t ? Array.from({ length: azims }, (_, k) => k * 360 / azims) : [0])) list.push([t, a]);
  for (const [i, [t, a]] of list.entries()) {
    const rd = render(tris, viewRotation(t, a), ppm, size, camDist);
    [rd.nx, rd.ny] = edgeNormals(rd);
    const [tcy, tcx, tr, area] = centroidRadius(rd.mask, size, size), f = C / size, coarse = new Uint8Array(C * C);
    for (let y = 0; y < C; y++) for (let x = 0; x < C; x++) {
      const sx = Math.round((x - (C - 1) / 2) / f + tcx), sy = Math.round((y - (C - 1) / 2) / f + tcy);
      if (sx >= 0 && sy >= 0 && sx < size && sy < size) coarse[y * C + x] = rd.mask[sy * size + sx];
    }
    views.push({ tilt: t, azim: a, rd, tcy, tcx, tr, coarse, coarseArea: count(coarse) });
    if (opts.onProgress) await opts.onProgress(i + 1, list.length);
  }
  return { views, size, ppm, coarse: C, camDist };
}

// ---------------------------------------------------------------- photo side

const LIN = Float32Array.from({ length: 1024 }, (_, i) => { const c = i / 4 / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });

// Part mask on a coloured mat: how much of the mat's own hue a pixel carries relative to its lightness.
// Lit and shadowed mat score the same, grey metal (highlights included) ~0; Otsu splits the two.
function segment(rgba, w, h) {
  const N = w * h, R = new Float32Array(N), G = new Float32Array(N), B = new Float32Array(N);
  for (let p = 0; p < N; p++) { R[p] = rgba[p * 4]; G[p] = rgba[p * 4 + 1]; B[p] = rgba[p * 4 + 2]; }
  const Rb = blur(R, w, h, 1.1, 2), Gb = blur(G, w, h, 1.1, 2), Bb = blur(B, w, h, 1.1, 2);
  const L = new Float32Array(N), A = new Float32Array(N), Bv = new Float32Array(N);
  const lin = c => { const i = Math.min(1023, Math.max(0, c * 4)), k = i | 0, f = i - k; return LIN[k] + f * (LIN[Math.min(1023, k + 1)] - LIN[k]); };
  const fl = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  for (let p = 0; p < N; p++) {
    const r = lin(Rb[p]), g = lin(Gb[p]), b = lin(Bb[p]);
    const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, Y = 0.2126 * r + 0.7152 * g + 0.0722 * b, Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    const fx = fl(X), fy = fl(Y), fz = fl(Z);
    L[p] = (116 * fy - 16) * 2.55; A[p] = 500 * (fx - fy); Bv[p] = 200 * (fy - fz);  // L scaled like OpenCV 8-bit
  }
  const border = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x < 6 || y < 6 || x >= w - 6 || y >= h - 6) border.push(y * w + x);
  let ua = quantile(border.map(p => A[p]), 0.5), ub = quantile(border.map(p => Bv[p]), 0.5);
  const ul = Math.hypot(ua, ub) + 1e-6; ua /= ul; ub /= ul;
  const s = new Float32Array(N);
  for (let p = 0; p < N; p++) s[p] = (A[p] * ua + Bv[p] * ub) / (L[p] + 20);
  const sMat = quantile(border.map(p => s[p]), 0.5);
  let mx = 0;
  const d = new Float32Array(N);
  for (let p = 0; p < N; p++) { d[p] = Math.max(0, sMat - s[p]); mx = Math.max(mx, d[p]); }
  const u8 = new Uint8Array(N);
  for (let p = 0; p < N; p++) u8[p] = Math.round(255 * d[p] / (mx || 1));
  const t = otsu(u8);
  let m = u8.map(v => v > t ? 1 : 0);
  m = close(open(m, w, h, 3), w, h, 5);
  const cc = components(m, w, h, true);
  if (cc.n < 1) return new Uint8Array(N);
  let k = 1;
  for (let j = 2; j <= cc.n; j++) if (cc.area[j] > cc.area[k]) k = j;
  const part = cc.lab.map(l => l === k ? 1 : 0);
  // small background specks inside the part are noise; real holes are kept
  const hc = components(enclosedBackground(part, w, h), w, h, false);
  for (let p = 0; p < N; p++) if (hc.lab[p] && hc.area[hc.lab[p]] < 12) part[p] = 1;
  return part;
}

// Affine maps as [a, b, c, d, e, f]: x' = a x + b y + c, y' = d x + e y + f
const apply = (M, x, y) => [M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]];
function invert(M) {
  const det = M[0] * M[4] - M[1] * M[3];
  const a = M[4] / det, b = -M[1] / det, d = -M[3] / det, e = M[0] / det;
  return [a, b, -(a * M[2] + b * M[5]), d, e, -(d * M[2] + e * M[5])];
}
function sample(img, w, h, x, y) {  // bilinear, 0 outside
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const at = (xx, yy) => (xx < 0 || yy < 0 || xx >= w || yy >= h) ? 0 : img[yy * w + xx];
  return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}
// Pull a photo-frame image into the template frame through W (template -> photo)
function warp(img, w, h, W, S, nearest) {
  const out = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const [u, v] = apply(W, x, y);
    out[y * S + x] = nearest ? ((u < 0 || v < 0 || u >= w - 0.5 || v >= h - 0.5) ? 0 : img[Math.round(v) * w + Math.round(u)]) : sample(img, w, h, u, v);
  }
  return out;
}

// Coarse: every template view x in-plane rotation (hole-aware silhouettes, sampled points). Fine: Lucas-Kanade affine
// on blurred masks. Returns W, the template -> photo affine.
function align(prof, mask, w, h) {
  const [cy, cx, r, areaP] = centroidRadius(mask, w, h), C = prof.coarse, S = prof.size, f = C / S;
  const pts = [], step = Math.max(1, Math.floor(Math.sqrt(areaP / 3000)));
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) if (mask[y * w + x]) pts.push(x - cx, y - cy);
  const np = pts.length / 2;
  const perView = [];
  for (let vi = 0; vi < prof.views.length; vi++) {
    const v = prof.views[vi], scale = f * v.tr / r, aP = areaP * scale * scale;
    let vb = null;
    for (let ang = 0; ang < 360; ang += 2) {
      const th = ang * Math.PI / 180, a = scale * Math.cos(th), b = scale * Math.sin(th), h2 = (C - 1) / 2;
      let hit = 0;
      for (let i = 0; i < pts.length; i += 2) {
        const x = Math.round(a * pts[i] + b * pts[i + 1] + h2), y = Math.round(-b * pts[i] + a * pts[i + 1] + h2);
        if (x >= 0 && y >= 0 && x < C && y < C && v.coarse[y * C + x]) hit++;
      }
      const inter = hit / np * aP, iou = inter / (aP + v.coarseArea - inter);
      if (!vb || iou > vb.iou) vb = { iou, vi, ang, scale: v.tr / r };
    }
    perView.push(vb);
  }
  perView.sort((p, q) => q.iou - p.iou);
  // The outline barely changes with a few degrees of tilt, but what shows through deep holes does: fine-align the
  // views whose outline matches about equally well and keep the one whose see-through holes agree with the photo.
  const cands = perView.filter(c => c.iou >= perView[0].iou - 0.015).slice(0, 3);
  let best = null;
  for (const c of cands) {
    const W = refine(prof, c, mask, w, h, cx, cy), v = prof.views[c.vi], tm = v.rd.mask, th = v.rd.holes;
    const pm = warp(mask, w, h, W, S, true).map(x => x > 0.5 ? 1 : 0), ph = enclosedBackground(pm, S, S);
    let i1 = 0, u1 = 0, i2 = 0, u2 = 0;
    for (let p = 0; p < S * S; p++) { i1 += pm[p] && tm[p]; u1 += pm[p] || tm[p]; i2 += ph[p] && th[p]; u2 += ph[p] || th[p]; }
    const score = i1 / Math.max(u1, 1) + (u2 ? i2 / u2 : 1);
    if (!best || score > best.score) best = { score, W, vi: c.vi, iou: c.iou, scale: c.scale };
  }
  return { view: best.vi, W: best.W, coarseIou: best.iou, scale: best.scale };
}

function refine(prof, c, mask, w, h, cx, cy) {
  const S = prof.size, v = prof.views[c.vi], th = c.ang * Math.PI / 180, a = c.scale * Math.cos(th), b = c.scale * Math.sin(th);
  // photo -> template: rotate/scale about the photo centroid onto the template centroid
  const A = [a, b, v.tcx - a * cx - b * cy, -b, a, v.tcy + b * cx - a * cy];
  const W = invert(A);
  const T = blur(Float32Array.from(v.rd.mask), S, S, 2.0), I = blur(Float32Array.from(mask), w, h, 2.0 / c.scale);
  const [Ix, Iy] = sobel(I, w, h);
  for (let it = 0; it < 30; it++) {
    const H = new Float64Array(36), g = new Float64Array(6), stp = it < 12 ? 3 : 2;
    for (let y = 0; y < S; y += stp) for (let x = 0; x < S; x += stp) {
      const t = T[y * S + x], [u, vv] = apply(W, x, y), iw = sample(I, w, h, u, vv);
      if ((t < 0.01 && iw < 0.01) || (t > 0.99 && iw > 0.99)) continue;
      const gx = sample(Ix, w, h, u, vv) / 8, gy = sample(Iy, w, h, u, vv) / 8, e = iw - t;
      const J = [gx * x, gx * y, gx, gy * x, gy * y, gy];
      for (let i = 0; i < 6; i++) { g[i] += J[i] * e; for (let j = 0; j < 6; j++) H[i * 6 + j] += J[i] * J[j]; }
    }
    const dp = solve6(H, g);
    if (!dp) break;
    for (let i = 0; i < 6; i++) W[i] -= dp[i];
    if (Math.abs(dp[2]) + Math.abs(dp[5]) < 0.01 && Math.abs(dp[0]) + Math.abs(dp[1]) + Math.abs(dp[3]) + Math.abs(dp[4]) < 1e-4) break;
  }
  return W;
}
function solve6(H, g) {  // Gaussian elimination with partial pivoting
  const n = 6, M = Array.from({ length: n }, (_, i) => [...H.slice(i * n, i * n + n), g[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j]; }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

// ---------------------------------------------------------------- measurements and decision

function measure(prof, rgba, w, h) {
  const mask = segment(rgba, w, h);
  if (count(mask) < 500) return { noPart: true };
  const al = align(prof, mask, w, h), S = prof.size, N = S * S, ppm = prof.ppm, mm2 = 1 / (ppm * ppm);
  const rd = prof.views[al.view].rd, tm = rd.mask, th = rd.holes, te = rd.edges;
  const pm = warp(mask, w, h, al.W, S, false).map(v => v > 0.5 ? 1 : 0);
  const gray = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) gray[p] = 0.299 * rgba[p * 4] + 0.587 * rgba[p * 4 + 1] + 0.114 * rgba[p * 4 + 2];
  const grayT = warp(gray, w, h, al.W, S, false);
  // edge evidence across the CAD's edge direction vs. edge strength elsewhere on the part
  const [gx, gy] = sobel(blur(grayT, S, S, 1.2), S, S), Gm = new Float32Array(N);
  for (let p = 0; p < N; p++) Gm[p] = Math.hypot(gx[p], gy[p]);
  const inner = erode(and(pm, tm), S, S, 7), s95 = quantile(pick(Gm, inner), 0.95) + 1e-6;
  const proj = new Float32Array(N);
  for (let p = 0; p < N; p++) proj[p] = Math.abs(gx[p] * rd.nx[p] + gy[p] * rd.ny[p]) / s95;
  const P = dilate(proj, S, S, 5), Gd = dilate(Gm.map(v => v / s95), S, S, 5);
  const nearOutline = dilate(rd.outline, S, S, 7), interior = and(te, not(nearOutline), not(dilate(th, S, S, 7)));
  const far = not(dilate(te, S, S, 9)), body = erode(and(tm, not(th)), S, S, 9);
  const on = quantile(pick(P, interior), 0.5), off = quantile(pick(Gd, and(body, far)), 0.5) * 2 / Math.PI;
  let inter = 0, uni = 0;
  for (let p = 0; p < N; p++) { inter += pm[p] && tm[p]; uni += pm[p] || tm[p]; }
  const st = { iou: inter / Math.max(uni, 1), featureRatio: on / Math.max(off, 1e-6), view: al.view, tilt: prof.views[al.view].tilt, W: al.W, size: S, ppm,
    partMm2: count(tm) * mm2 };
  const tol = Math.max(1, Math.round(ppm)), kt = 2 * tol + 1, ml = rd.mouth, mouth = ml.map(v => v ? 1 : 0);
  // designed through-holes: is background showing anywhere in the hole's mouth, about as much as the CAD predicts?
  const hc = components(th, S, S, false);
  st.holes = [];
  for (let j = 1; j <= hc.n; j++) {
    let see = 0, clear = 0, mouthPx = 0;
    for (let p = 0; p < N; p++) { if (hc.lab[p] === j) see++; if (ml[p] === j) { mouthPx++; if (!pm[p]) clear++; } }
    st.holes.push({ label: j, areaMm2: see * mm2, mouthMm2: mouthPx * mm2, open: Math.min(1, clear / Math.max(see, 1)) });
  }
  st.holeLabels = ml;
  const photoHoles = enclosedBackground(pm, S, S), expectedBg = dilate(or(mouth, not(tm)), S, S, kt);
  // part-vs-CAD disagreement away from the outline band and the hole mouths, as a share of the part's area
  const tmArea = count(tm), freeOfHoles = not(dilate(mouth, S, S, kt));
  for (const mmBand of [1.0, 1.5, 2.0]) {
    const kb = 2 * Math.max(1, Math.round(mmBand * ppm)) + 1, away = and(not(dilate(rd.outline, S, S, kb)), freeOfHoles);
    let bad = 0;
    for (let p = 0; p < N; p++) bad += away[p] && (pm[p] !== tm[p]);
    st["mismatch" + mmBand * 10] = bad / Math.max(tmArea, 1);
  }
  const comp = (m, conn8) => { const c = components(m, S, S, conn8); return { lab: c.lab, items: c.area.slice(1).map((a, i) => ({ label: i + 1, areaMm2: a * mm2 })) }; };
  st.extra = comp(and(photoHoles, not(expectedBg)), false);
  const band = dilate(rd.outline, S, S, kt);
  st.missing = comp(open(and(tm, not(th), not(pm), not(band), not(photoHoles)), S, S, 3), true);
  st.excess = comp(open(and(pm, not(tm), not(band)), S, S, 3), true);
  return st;
}

const DEFAULTS = { iouMin: 0.90, featureRatioMin: 1.45, patternMismatch: 0.5, holeOpenMin: 0.5, extraHoleMm2: 2.0, outlineMm2: 6.0, seeThroughMin: 0.45 };

// Shape of every labelled component: area, ellipse-equivalent length/width (4 sigma along the principal axes) and how
// much of that ellipse it fills. Units are pixels; callers convert with ppm.
function shapes(lab, w) {
  const acc = {};
  for (let p = 0; p < lab.length; p++) {
    const l = lab[p]; if (!l) continue;
    const x = p % w, y = (p / w) | 0, a = acc[l] ||= [0, 0, 0, 0, 0, 0];
    a[0]++; a[1] += x; a[2] += y; a[3] += x * x; a[4] += y * y; a[5] += x * y;
  }
  const out = {};
  for (const [l, [n, sx, sy, sxx, syy, sxy]] of Object.entries(acc)) {
    const mx = sx / n, my = sy / n, cxx = sxx / n - mx * mx + 1 / 12, cyy = syy / n - my * my + 1 / 12, cxy = sxy / n - mx * my;
    const tr = (cxx + cyy) / 2, det = Math.sqrt(Math.max(0, tr * tr - (cxx * cyy - cxy * cxy)));
    const len = 4 * Math.sqrt(tr + det), wid = 4 * Math.sqrt(Math.max(tr - det, 1 / 12));
    out[l] = { n, cx: mx, cy: my, len, wid, fill: n / (Math.PI / 4 * len * wid) };
  }
  return out;
}

// Defect type from where the deviation sits (hole / outline) and its shape in mm. Foundry names, sizes in mm.
function typeFinding(kind, sh, ppm, extra) {
  const len = sh.len / ppm, wid = sh.wid / ppm, dia = 2 * Math.sqrt(sh.n / Math.PI) / ppm, el = sh.len / sh.wid;
  if (kind === "unexpected_hole") {
    // a narrow crack or a ragged void only shows the mat through in fragments; sh.parts counts them
    if (el >= 3 && wid < 3) return { defect: "through crack", size: `~${len.toFixed(0)} mm long` };
    if (sh.parts > 1 || (sh.fill < 0.6 && dia >= 3)) return { defect: "shrinkage cavity (through)", size: `~${len.toFixed(1)} × ${wid.toFixed(1)} mm` };
    return { defect: dia < 2 ? "pinhole (through)" : "blowhole (through)", size: `Ø ${dia.toFixed(1)} mm` };
  }
  if (kind === "blocked_hole")
    return extra.open < 0.15 ? { defect: "hole not formed (core missing or broken)", size: `Ø ${dia.toFixed(1)} mm hole` }
      : { defect: "partly blocked hole (sand or flash in it)", size: `${Math.round(100 * (1 - extra.open))}% blocked` };
  if (kind === "missing_material")
    return sh.n / (ppm * ppm) >= 150 || len >= 25 ? { defect: "broken or unfilled section (misrun)", size: `~${len.toFixed(0)} × ${wid.toFixed(0)} mm` }
      : { defect: "chipped edge", size: `~${len.toFixed(1)} × ${wid.toFixed(1)} mm` };
  if (kind === "excess_material")
    return el >= 4 && wid < 8 ? { defect: "flash / fin", size: `~${len.toFixed(0)} mm long, sticks out ~${wid.toFixed(1)} mm` }
      : { defect: "extra metal (swell or lump)", size: `~${len.toFixed(1)} × ${wid.toFixed(1)} mm` };
  return { defect: kind.replace(/_/g, " "), size: "" };
}

// Verdict plus typed findings; `mask` marks finding pixels in the template frame (for drawing)
function decide(st, cfg) {
  cfg = { ...DEFAULTS, ...cfg };
  if (st.noPart) return { verdict: "NO PART", findings: [], mask: null, unchecked: 0 };
  const ppm = st.ppm || 1;
  const S = st.size, mask = new Uint8Array(S * S), F = [];
  // A deep or off-axis hole may show almost no background from this camera position; it can't be checked for "blocked"
  const checkable = st.holes.filter(hh => hh.areaMm2 >= cfg.seeThroughMin * (hh.mouthMm2 || hh.areaMm2));
  const blocked = checkable.filter(hh => hh.open < cfg.holeOpenMin), unchecked = st.holes.length - checkable.length;
  const holeArea = st.holes.reduce((s, hh) => s + hh.areaMm2, 0) || 1;
  const extra = st.extra.items.filter(c => c.areaMm2 >= cfg.extraHoleMm2);
  const extraArea = extra.reduce((s, c) => s + c.areaMm2, 0);
  const patternOff = blocked.length >= Math.max(2, Math.ceil(cfg.patternMismatch * checkable.length)) || extraArea > cfg.patternMismatch * holeArea;
  if (st.iou < cfg.iouMin || st.featureRatio < cfg.featureRatioMin || patternOff) {
    const why = st.iou < cfg.iouMin ? "outline does not match" : patternOff ? "hole pattern does not match" : "features do not match";
    return { verdict: "WRONG PART", findings: [{ type: "mismatch", text: why }], mask: null, unchecked };
  }
  const mark = (lab, label) => { for (let p = 0; p < lab.length; p++) if (lab[p] === label) mask[p] = 1; };
  const add = (type, lab, sh, c, note, ex) => {
    const t = typeFinding(type, sh[c.label], ppm, ex);
    F.push({ type, defect: t.defect, size: t.size, areaMm2: +c.areaMm2.toFixed(1), text: `${t.defect}, ${t.size} (${note})` });
    mark(lab, c.label);
  };
  const mouthSh = blocked.length ? shapes(st.holeLabels, S) : {};
  for (const hh of blocked) add("blocked_hole", st.holeLabels, mouthSh, { label: hh.label, areaMm2: hh.mouthMm2 || hh.areaMm2 }, "designed hole", hh);
  // Group see-through fragments within 2.5 mm of a flagged opening; the group's shape decides the type
  if (extra.length) {
    const keep = new Set(st.extra.items.filter(c => c.areaMm2 >= 0.3).map(c => c.label)), frag = st.extra.lab.map(l => keep.has(l) ? 1 : 0);
    const k = 2 * Math.max(1, Math.round(2.5 * ppm)) + 1, cl = components(dilate(frag, S, S, k), S, S, true).lab;
    const lab = new Int32Array(S * S), parts = {}, area = {};
    for (let p = 0; p < S * S; p++) if (frag[p]) lab[p] = cl[p];
    for (const c of st.extra.items) if (keep.has(c.label)) {
      const p0 = st.extra.lab.indexOf(c.label), g = cl[p0];
      parts[g] = (parts[g] || 0) + 1; area[g] = (area[g] || 0) + c.areaMm2;
    }
    const sh = shapes(lab, S), done = new Set();
    for (const c of extra) {
      const g = cl[st.extra.lab.indexOf(c.label)];
      if (done.has(g)) continue;
      done.add(g); sh[g].parts = parts[g];
      add("unexpected_hole", lab, sh, { label: g, areaMm2: area[g] }, "hole not in the CAD");
    }
  }
  // outline tolerance scales down for small parts (0.2% of the part's area, 2 mm² at least, never above outlineMm2)
  const outlineMin = Math.min(cfg.outlineMm2, Math.max(2, 0.002 * (st.partMm2 || Infinity)));
  const miss = st.missing.items.filter(c => c.areaMm2 >= outlineMin), exc = st.excess.items.filter(c => c.areaMm2 >= outlineMin);
  const miSh = miss.length ? shapes(st.missing.lab, S) : {}, xsSh = exc.length ? shapes(st.excess.lab, S) : {};
  for (const c of miss) add("missing_material", st.missing.lab, miSh, c, `${c.areaMm2.toFixed(0)} mm² missing vs CAD`);
  for (const c of exc) {
    // extra metal sitting in a designed hole's mouth is a hole that was not cored, even where the hole can't be seen through
    let on = 0, n = 0;
    for (let p = 0; p < S * S; p++) if (st.excess.lab[p] === c.label) { n++; on += st.holeLabels[p] ? 1 : 0; }
    if (on >= 0.3 * n) add("blocked_hole", st.excess.lab, xsSh, c, `${c.areaMm2.toFixed(0)} mm² of metal in a designed hole`, { open: 0 });
    else add("excess_material", st.excess.lab, xsSh, c, `${c.areaMm2.toFixed(0)} mm² extra vs CAD`);
  }
  F.sort((a, b) => b.areaMm2 - a.areaMm2);  // the biggest deviation leads
  return { verdict: F.length ? "REJECT" : "PASS", findings: F, mask, unchecked };
}

// ---------------------------------------------------------------- reference-images tab: what kind of defect

// A good part kept at enrolment for typing: grey image, part mask (null without a coloured mat), centre and size
function surfaceRef(rgba, w, h) {
  const N = w * h, G = new Uint8Array(N);
  for (let p = 0; p < N; p++) G[p] = Math.round(0.299 * rgba[p * 4] + 0.587 * rgba[p * 4 + 1] + 0.114 * rgba[p * 4 + 2]);
  let M = segment(rgba, w, h), n = count(M), border = 0;
  for (let x = 0; x < w; x++) border += M[x] + M[(h - 1) * w + x];
  for (let y = 0; y < h; y++) border += M[y * w] + M[y * w + w - 1];
  if (n < 0.03 * N || n > 0.85 * N || border > 0.1 * 2 * (w + h)) M = null;  // no usable part/background split
  let cx = w / 2, cy = h / 2, r = 0.42 * Math.min(w, h);
  if (M) {
    let sx = 0, sy = 0;
    for (let p = 0; p < N; p++) if (M[p]) { sx += p % w; sy += (p / w) | 0; }
    cx = sx / n; cy = sy / n;
    // size: median distance of the outer outline from the centre, so a chip or a fin barely moves it
    const inner = enclosedBackground(M, w, h), ds = [];
    for (let p = 0; p < N; p++) {
      if (!M[p]) continue;
      const x = p % w, y = (p / w) | 0;
      if (x && y && x < w - 1 && y < h - 1 && (!M[p - 1] && !inner[p - 1] || !M[p + 1] && !inner[p + 1] || !M[p - w] && !inner[p - w] || !M[p + w] && !inner[p + w]))
        ds.push(Math.hypot(x - cx, y - cy));
    }
    r = quantile(ds, 0.5) || r;
  }
  return { w, h, G, M, cx, cy, r };
}

// Ring-normalised polar samples (angle-major) for the rotation search: grey rings, then mask rings weighted up
// so through-holes, which often break a part's symmetry, decide between near-equal rotations
const NA = 360, NR = 40, NC = 2 * NR;
function polar(ref) {
  const P = new Float32Array(NA * NC);
  for (let c = 0; c < 2; c++) for (let t = 0; t < NR; t++) {
    let s = 0, q = 0;
    for (let a = 0; a < NA; a++) {
      const th = 2 * Math.PI * a / NA, rr = ref.r * (0.05 + 1.0 * t / NR), x = ref.cx + rr * Math.cos(th), y = ref.cy + rr * Math.sin(th);
      const v = c ? (ref.M ? 255 * (ref.M[Math.min(ref.h - 1, Math.max(0, Math.round(y))) * ref.w + Math.min(ref.w - 1, Math.max(0, Math.round(x)))] || 0) : 0) : sample(ref.G, ref.w, ref.h, x, y);
      P[a * NC + c * NR + t] = v; s += v; q += v * v;
    }
    const m = s / NA, sd = Math.sqrt(Math.max(q / NA - m * m, 0)) + 8;
    for (let a = 0; a < NA; a++) P[a * NC + c * NR + t] = (P[a * NC + c * NR + t] - m) / sd * (c ? 2 : 1);
  }
  return P;
}

// Rotation (radians) of ref onto test and the normalised match score
function rotationOf(Pt, Pr) {
  let best = -Infinity, bk = 0, nt = 0, nr = 0;
  for (let i = 0; i < Pt.length; i++) { nt += Pt[i] * Pt[i]; nr += Pr[i] * Pr[i]; }
  for (let k = 0; k < NA; k++) {
    let s = 0;
    for (let a = 0; a < NA; a++) { const o = a * NC, q = ((a + k) % NA) * NC; for (let t = 0; t < NC; t++) s += Pt[o + t] * Pr[q + t]; }
    if (s > best) { best = s; bk = k; }
  }
  return { rot: 2 * Math.PI * bk / NA, score: best / Math.sqrt(nt * nr + 1e-9) };
}

// Ref grey and mask resampled into the test photo's frame (rotation about the centres, scale from part size)
function warpRef(ref, test, rot, dx, dy) {
  const { w, h } = test, N = w * h, G = new Float32Array(N), M = new Uint8Array(N), s = ref.r / test.r, c = Math.cos(rot), sn = Math.sin(rot);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const u = x - test.cx - dx, v = y - test.cy - dy, X = ref.cx + s * (c * u - sn * v), Y = ref.cy + s * (sn * u + c * v), p = y * w + x;
    G[p] = sample(ref.G, ref.w, ref.h, X, Y);
    const xi = Math.round(X), yi = Math.round(Y);
    M[p] = ref.M && xi >= 0 && yi >= 0 && xi < ref.w && yi < ref.h ? ref.M[yi * ref.w + xi] : 0;
  }
  return { G, M };
}

// Likely defect type for the hottest areas of a patch anomaly map d (H x W cells over the w x h photo), read from
// how the photo differs from the best-matching good reference inside each area.
function typeSurface(rgba, w, h, d, H, W, thr, refs) {
  if (!refs || !refs.length) return [];
  const N = w * h, test = surfaceRef(rgba, w, h), Pt = polar(test);
  // best reference by rotation match, then a small shift search on the masks
  let bestRef = null, best = { score: -Infinity };
  for (const ref of refs) {
    if (!!ref.M !== !!test.M) continue;
    const r = rotationOf(Pt, ref.P ||= polar(ref));
    if (r.score > best.score) { best = r; bestRef = ref; }
  }
  if (!bestRef) return [];
  let dx = 0, dy = 0;
  if (test.M) {
    let top = -1;
    for (let oy = -4; oy <= 4; oy += 2) for (let ox = -4; ox <= 4; ox += 2) {
      const wr = warpRef(bestRef, { ...test, w: w >> 2, h: h >> 2, cx: test.cx / 4, cy: test.cy / 4, r: test.r / 4 }, best.rot, ox / 4, oy / 4);
      let a = 0;
      for (let y = 0; y < h >> 2; y++) for (let x = 0; x < w >> 2; x++) { const tm = test.M[(4 * y) * w + 4 * x]; a += tm === wr.M[y * (w >> 2) + x] ? 1 : 0; }
      if (a > top) { top = a; dx = ox; dy = oy; }
    }
  }
  const R = warpRef(bestRef, test, best.rot, dx, dy), Gt = blur(Float32Array.from(test.G), w, h, 1.2), Gr = blur(R.G, w, h, 1.2);
  const Mt = test.M || new Uint8Array(N).fill(1), Mr = test.M ? R.M : Mt;
  // match brightness on the shared part area, then signed differences
  const both = erode(and(Mt, Mr), w, h, 5), vt = pick(Gt, both), vr = pick(Gr, both);
  const mt = quantile(vt, 0.5), mr = quantile(vr, 0.5), st_ = quantile(vt.map(v => Math.abs(v - mt)), 0.5) + 1, sr = quantile(vr.map(v => Math.abs(v - mr)), 0.5) + 1;
  const diff = new Float32Array(N);
  for (let p = 0; p < N; p++) diff[p] = both[p] ? (Gt[p] - mt) / st_ - (Gr[p] - mr) / sr : 0;
  const noise = quantile(pick(diff, both).map(Math.abs), 0.9) + 1e-6;
  const newDark = open(both.map((v, p) => v && diff[p] < -2.2 * noise ? 1 : 0), w, h, 2), lostDark = open(both.map((v, p) => v && diff[p] > 2.2 * noise ? 1 : 0), w, h, 2);
  const lost = open(and(Mr, not(Mt)), w, h, 5), gained = open(and(Mt, not(Mr)), w, h, 5);
  const holesT = test.M ? enclosedBackground(Mt, w, h) : new Uint8Array(N), holesR = test.M ? dilate(enclosedBackground(Mr, w, h), w, h, 5) : new Uint8Array(N);
  const partPx = count(Mt), out = [];
  const add = (defect, why, cx, cy, score) => out.push({ type: "surface", defect, why, cx, cy, score, text: `likely ${defect} (${why})` });
  // areas: connected hot cells around the strongest peaks
  let peak = 0;
  for (let j = 1; j < d.length; j++) if (d[j] > d[peak]) peak = j;
  const hot = new Uint8Array(H * W);
  for (let j = 0; j < d.length; j++) hot[j] = d[j] >= Math.max(0.85 * thr, 0.8 * d[peak]) ? 1 : 0;
  const cc = components(hot, W, H, true), bestCell = {};
  for (let j = 0; j < d.length; j++) { const l = cc.lab[j]; if (l && (!bestCell[l] || d[j] > d[bestCell[l]])) bestCell[l] = j; }
  const areas = Object.entries(bestCell).map(([l, j]) => ({ l: +l, j, v: d[j] })).filter(a => a.v >= 0.92 * d[peak] || a.l === cc.lab[peak])
    .sort((a, b) => b.v - a.v).slice(0, 3);
  const cw = w / W, ch = h / H;
  for (const a of areas) {
    const Rc = new Uint8Array(N);
    for (let j = 0; j < d.length; j++) if (cc.lab[j] === a.l) {
      const x0 = Math.floor((j % W) * cw), y0 = Math.floor(((j / W) | 0) * ch);
      for (let y = y0; y < Math.min(h, Math.ceil(y0 + ch)); y++) for (let x = x0; x < Math.min(w, Math.ceil(x0 + cw)); x++) Rc[y * w + x] = 1;
    }
    const Rd = dilate(Rc, w, h, 2 * Math.round(cw) + 1), cx = ((a.j % W) + 0.5) / W, cy = (((a.j / W) | 0) + 0.5) / H;
    const inR = m => { let n = 0; for (let p = 0; p < N; p++) n += m[p] && Rd[p] ? 1 : 0; return n; };
    const nLost = inR(lost), nGained = inR(gained), minPx = Math.max(12, partPx / 2000);
    if (nLost >= minPx && nLost >= nGained) {
      const opening = and(lost, holesT, Rd), nOpen = count(opening);
      if (nOpen >= 0.5 * nLost) {
        const sh = shapes(components(opening, w, h, true).lab, w), big = Object.values(sh).sort((x, y) => y.n - x.n)[0];
        if (big.len / big.wid >= 3) add("through crack", "a slit shows the background where good parts are solid", cx, cy, a.v);
        else add("blowhole (through)", "a hole where good parts are solid", cx, cy, a.v);
      } else add(nLost >= 0.012 * partPx ? "broken or unfilled section (misrun)" : "chipped edge", "material missing at the outline", cx, cy, a.v);
      continue;
    }
    if (nGained >= minPx) {
      if (inR(and(gained, holesR)) >= 0.5 * nGained) add("blocked hole", "metal where good parts have a hole", cx, cy, a.v);
      else {
        const sh = Object.values(shapes(components(and(gained, Rd), w, h, true).lab, w)).sort((x, y) => y.n - x.n)[0];
        add(sh.len / sh.wid >= 4 ? "flash / fin" : "extra metal (swell or lump)", "material beyond the good part's outline", cx, cy, a.v);
      }
      continue;
    }
    const nd = and(newDark, Rd), nDark = count(nd), nLostDark = inR(lostDark);
    if (nLostDark >= Math.max(40, 0.8 * nDark)) { add("missing or deformed feature", "a feature the good parts have looks different or is gone", cx, cy, a.v); continue; }
    if (nDark >= 8) {
      // fragments of one mark within 3 px are measured together, so a crack broken up by texture still reads as a line
      const grp = components(dilate(nd, w, h, 5), w, h, true).lab, gl = new Int32Array(N);
      for (let p = 0; p < N; p++) if (nd[p]) gl[p] = grp[p];
      const sh = Object.values(shapes(gl, w)).filter(s => s.n >= 6), line = sh.find(s => s.len >= w / 30 && s.len / s.wid >= 3.5);
      const pits = sh.filter(s => s.len / s.wid < 2.5);
      if (line) add("crack", "a thin dark line good parts don't have", cx, cy, a.v);
      else if (pits.length >= 3) add("gas porosity", `${pits.length} small dark pits`, cx, cy, a.v);
      else if (pits.length) add("blowhole / pit (surface)", "a dark pit good parts don't have", cx, cy, a.v);
      else add("surface defect", "darker marks than on good parts (inclusion, scab or sand)", cx, cy, a.v);
      continue;
    }
    add("surface anomaly", "texture unlike the good parts", cx, cy, a.v);
  }
  // a designed hole that is filled in, wherever the heat map peaked
  if (test.M && !out.some(f => f.defect === "blocked hole")) {
    const hr = components(enclosedBackground(Mr, w, h), w, h, false);
    for (let l = 1; l <= hr.n; l++) {
      if (hr.area[l] < 25) continue;
      let filled = 0, sx = 0, sy = 0;
      for (let p = 0; p < N; p++) if (hr.lab[p] === l) { if (Mt[p] && !dilate1(holesT, p, w, h)) filled++; sx += p % w; sy += (p / w) | 0; }
      if (filled >= 0.6 * hr.area[l]) add("blocked hole", "a hole every good part has is filled", sx / hr.area[l] / w, sy / hr.area[l] / h, 0);
    }
  }
  // a hole that is plainly filled or opened beats a texture-level guess
  const rank = f => /blocked|through|chipped|misrun|flash|extra metal/.test(f.defect) ? 0 : 1;
  return out.map((f, i) => [f, i]).sort((x, y) => rank(x[0]) - rank(y[0]) || x[1] - y[1]).map(x => x[0]);
}
const dilate1 = (m, p, w, h) => { const x = p % w, y = (p / w) | 0; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < w && yy < h && m[yy * w + xx]) return true; } return false; };

// RGBA overlay in photo coordinates: CAD edges red, designed holes blue, findings filled; plus the findings' centroid
function overlay(prof, st, findMask, w, h) {
  const out = new Uint8ClampedArray(w * h * 4), S = prof.size, rd = prof.views[st.view].rd, A = invert(st.W);
  let fx = 0, fy = 0, fn = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [u, v] = apply(A, x, y), tx = Math.round(u), ty = Math.round(v);
    if (tx < 0 || ty < 0 || tx >= S || ty >= S) continue;
    const q = ty * S + tx, o = (y * w + x) * 4;
    if (findMask && findMask[q]) { out.set([255, 40, 60, 170], o); fx += x; fy += y; fn++; }
    else if (rd.edges[q]) out.set([255, 70, 70, 230], o);
    else if (rd.holes[q]) out.set([40, 170, 255, 120], o);
  }
  return { rgba: out, centroid: fn ? [fx / fn / w, fy / fn / h] : null };
}

// Small depth-shaded preview of one face, for the face picker
function preview(tris, face, size) {
  const { tris: t } = prepare(tris, face);
  let r = 0;
  for (let i = 0; i < t.length; i += 3) r = Math.max(r, Math.hypot(t[i], t[i + 1]));
  const rd = render(t, viewRotation(0, 0), size * 0.45 / r, size, 0), out = new Uint8ClampedArray(size * size * 4);
  let lo = Infinity, hi = -Infinity;
  for (let p = 0; p < size * size; p++) if (rd.mask[p]) { lo = Math.min(lo, rd.depth[p]); hi = Math.max(hi, rd.depth[p]); }
  for (let p = 0; p < size * size; p++) {
    const v = rd.mask[p] ? 90 + 120 * (rd.depth[p] - lo) / ((hi - lo) || 1) : 0;
    out.set(rd.edges[p] ? [255, 120, 130, 255] : rd.mask[p] ? [v, v, v, 255] : [0, 0, 0, 0], p * 4);
  }
  return out;
}

// Web Worker mode: the page posts {id, cmd, ...}; profiles live in the worker so inspections never block the UI
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  const profiles = {};
  self.onmessage = async ({ data: m }) => {
    try {
      if (m.cmd === "profile") {
        profiles[m.name] = await buildProfile(prepare(m.tris, m.face).tris, { camDist: m.camDist,
          onProgress: (i, n) => self.postMessage({ id: m.id, progress: [i, n] }) });
        self.postMessage({ id: m.id, result: true });
      } else if (m.cmd === "inspect") {
        const prof = profiles[m.name];
        if (!prof) throw new Error("CAD part not prepared");
        const st = measure(prof, m.rgba, m.w, m.h), d = decide(st, m.cfg), ov = st.noPart ? null : overlay(prof, st, d.mask, m.w, m.h);
        self.postMessage({ id: m.id, result: { noPart: !!st.noPart, iou: st.iou, featureRatio: st.featureRatio, verdict: d.verdict, unchecked: d.unchecked,
          findings: d.findings, overlay: ov && ov.rgba, centroid: ov && ov.centroid } }, ov ? [ov.rgba.buffer] : []);
      } else if (m.cmd === "surface") {
        self.postMessage({ id: m.id, result: typeSurface(m.rgba, m.w, m.h, m.d, m.H, m.W, m.thr, m.refs) });
      } else if (m.cmd === "drop") { delete profiles[m.name]; self.postMessage({ id: m.id, result: true }); }
    } catch (err) { self.postMessage({ id: m.id, error: err.message || String(err) }); }
  };
}

const CAD = { segment, surfaceRef, typeSurface, loadCAD, parseSTL, parseOBJ, prepare, buildProfile, measure, decide, overlay, preview, FACES: Object.keys(FACES), DEFAULTS };
if (typeof module !== "undefined" && module.exports) module.exports = CAD; else root.CAD = CAD;
})(typeof self !== "undefined" ? self : this);
