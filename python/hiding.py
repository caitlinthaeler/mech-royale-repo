"""Map knowledge for hiding: building raster, terrain walkability, hide-spot candidates, line of sight, A* planner.

Everything here is precomputed in Cover.__init__ (the hello budget is 20 s) or runs inside a small
per-tick time slice, so a decision always fits the 50 ms budget.
"""
import heapq
import json
import math
import os
import sys
import time
from array import array

CELL = 2.0    # raster cell for solids (m)
PCELL = 3.0   # planner cell (m)
RAYS = 12     # rays per hide-spot exposure estimate
RAY_LEN = 64.0
SIGHT = 45.0  # proximity sensor range: anything closer than this sees us regardless of facing


def _pip(x, y, ring):
    inside = False
    n = len(ring) // 2
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Cover:
    def __init__(self, map_dir, log=None):
        self.ok = False
        self.log = log or (lambda *a: None)
        self.solid = set()   # 2 m cells blocked for movement (buildings, stadium)
        self.tall = set()    # 2 m cells that also block line of sight (>= 9 m tall)
        self.h = None
        self.cands = []      # (x, y, exposure 0..1, open_yaw)
        self._walk = {}
        self.search = None
        t0 = time.time()
        try:
            self._load(map_dir)
            self.ok = True
        except Exception as e:  # no map: callers fall back to open-field behaviour
            self.log("cover: map load failed:", repr(e))
        self.log("cover: ready ok=%s solid=%d cands=%d in %.2fs" % (self.ok, len(self.solid), len(self.cands), time.time() - t0))

    # ------------------------------------------------------------------ loading
    def _load(self, d):
        w = json.load(open(os.path.join(d, "v2/world.json")))
        b = w["bounds"]
        self.minx, self.maxx, self.miny, self.maxy = b["minX"], b["maxX"], b["minY"], b["maxY"]
        t = w["terrain"]
        a = array("h")
        with open(os.path.join(d, "v2/terrain.i16"), "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder == "big":
            a.byteswap()
        self.h, self.tnx, self.tny = a, t["nx"], t["ny"]
        self.tminx, self.tminy, self.tstep = t["minX"], t["minY"], t["step"]
        for p in w["prisms"]:
            if p["kind"] == "bridge":
                continue  # walkable underneath; never treated as a wall
            ring = p["ring"]
            ground = min(self.ground(ring[k], ring[k + 1]) for k in range(0, len(ring), 2))
            self._raster(p, p["top"] - ground >= 9.0)
        self._build_candidates()

    def _raster(self, p, is_tall):
        ring = p["ring"]
        holes = p.get("holes") or []
        xs, ys = ring[0::2], ring[1::2]
        for i in range(int(min(xs) // CELL), int(max(xs) // CELL) + 1):
            for j in range(int(min(ys) // CELL), int(max(ys) // CELL) + 1):
                cx, cy = (i + 0.5) * CELL, (j + 0.5) * CELL
                if _pip(cx, cy, ring) and not any(_pip(cx, cy, hh) for hh in holes):
                    self.solid.add((i, j))
                    if is_tall:
                        self.tall.add((i, j))

    def ground(self, x, y):
        i = int(round((x - self.tminx) / self.tstep))
        j = int(round((y - self.tminy) / self.tstep))
        i = 0 if i < 0 else (self.tnx - 1 if i >= self.tnx else i)
        j = 0 if j < 0 else (self.tny - 1 if j >= self.tny else j)
        return self.h[j * self.tnx + i] * 0.01

    # ------------------------------------------------------------------ queries
    def in_bounds(self, x, y):
        return self.minx + 3 < x < self.maxx - 3 and self.miny + 3 < y < self.maxy - 3

    def hit(self, x, y, pad=0, tall=False):
        s = self.tall if tall else self.solid
        i, j = int(x // CELL), int(y // CELL)
        if pad == 0:
            return (i, j) in s
        for di in range(-pad, pad + 1):
            for dj in range(-pad, pad + 1):
                if (i + di, j + dj) in s:
                    return True
        return False

    def los(self, ax, ay, bx, by, tall=True):
        """True when no tall building sits between the two points."""
        d = math.hypot(bx - ax, by - ay)
        n = int(d / CELL)
        if n < 1:
            return True
        sx, sy = (bx - ax) / n, (by - ay) / n
        for k in range(1, n):
            if self.hit(ax + sx * k, ay + sy * k, 0, tall):
                return False
        return True

    def free_len(self, x, y, ang, limit, tall=True, pad=0):
        c, s = math.cos(ang), math.sin(ang)
        r = 0.0
        while r < limit:
            r += CELL
            if self.hit(x + c * r, y + s * r, pad, tall):
                return r - CELL
        return limit

    def clear_move(self, x, y, ang, dist):
        return self.free_len(x, y, ang, dist, tall=False, pad=1)

    def walkable(self, x, y):
        """Planner-cell test: not near a wall, not on a cliff or steep slope, inside the map."""
        key = (int(x // PCELL), int(y // PCELL))
        v = self._walk.get(key)
        if v is None:
            cx, cy = (key[0] + 0.5) * PCELL, (key[1] + 0.5) * PCELL
            v = self.in_bounds(cx, cy) and not self.hit(cx, cy, 1, False)
            if v:
                g = self.ground(cx, cy)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if abs(self.ground(cx + dx, cy + dy) - g) > 1.1:
                        v = False
                        break
            self._walk[key] = v
        return v

    # ------------------------------------------------------------------ hide spots
    def _build_candidates(self):
        step = 8.0
        angs = [k * 2 * math.pi / RAYS for k in range(RAYS)]
        out = []
        x = self.minx + 6
        while x < self.maxx - 6:
            y = self.miny + 6
            while y < self.maxy - 6:
                if not self.hit(x, y, 1, False) and self._near_tall(x, y):
                    tot, best, byaw = 0.0, -1.0, 0.0
                    for a in angs:
                        f = self.free_len(x, y, a, RAY_LEN, True)
                        tot += f
                        if f > best:
                            best, byaw = f, a
                    exp = tot / (RAYS * RAY_LEN)
                    if exp < 0.6 and self.walkable(x, y):
                        out.append((x, y, exp, byaw))
                y += step
            x += step
        out.sort(key=lambda c: c[2])
        self.cands = out[:5000]

    def _near_tall(self, x, y):
        for k in range(8):
            a = k * math.pi / 4
            if self.hit(x + math.cos(a) * 10, y + math.sin(a) * 10, 0, True):
                return True
        return False

    def choose_spot(self, me, center, radius, dangers, rng, avoid=None):
        """Best hide spot deep inside the circle (center, radius): enclosed, close, away from heard threats."""
        limit = max(0.0, radius * 0.85 - 8.0)
        best, bs = None, 1e18
        for (x, y, exp, yaw) in self.cands:
            dc = math.hypot(x - center[0], y - center[1])
            if dc > limit:
                continue
            s = exp * 120.0 + math.hypot(x - me[0], y - me[1]) * 0.12 + rng.random() * 6.0
            for (dx, dy, w) in dangers:
                d = math.hypot(x - dx, y - dy)
                if d < 110:
                    s += (110 - d) * w
            if avoid is not None:
                d = math.hypot(x - avoid[0], y - avoid[1])
                if d < 60:
                    s += (60 - d) * 1.5
            if s < bs:
                bs, best = s, (x, y, exp, yaw)
        return best

    def concealed_from(self, enemy, me, limit_circle, rng, radius=26.0):
        """Nearest point (within `radius` of me) the enemy has no line of sight to."""
        best, bs = None, 1e18
        for k in range(16):
            a = k * math.pi / 8
            for r in (8.0, 16.0, radius):
                x, y = me[0] + math.cos(a) * r, me[1] + math.sin(a) * r
                if limit_circle is not None and math.hypot(x - limit_circle[0], y - limit_circle[1]) > limit_circle[2]:
                    continue
                if not self.walkable(x, y) or self.los(enemy[0], enemy[1], x, y):
                    continue
                s = r - 0.15 * math.hypot(x - enemy[0], y - enemy[1]) + rng.random()
                if s < bs:
                    bs, best = s, (x, y)
        return best

    # ------------------------------------------------------------------ A* in time slices
    def _cell(self, x, y):
        return (int(x // PCELL), int(y // PCELL))

    def _center(self, c):
        return ((c[0] + 0.5) * PCELL, (c[1] + 0.5) * PCELL)

    def _nearest_walkable(self, c):
        if self.walkable(*self._center(c)):
            return c
        for r in range(1, 8):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) == r:
                        cc = (c[0] + di, c[1] + dj)
                        if self.walkable(*self._center(cc)):
                            return cc
        return None

    def start_path(self, a, b):
        s, g = self._nearest_walkable(self._cell(*a)), self._nearest_walkable(self._cell(*b))
        if s is None or g is None:
            self.search = None
            return
        self.search = {"goal": g, "target": b, "open": [(0.0, 0.0, s)], "g": {s: 0.0}, "from": {}, "done": None, "n": 0}

    def step_path(self, budget_s=0.012):
        """Advance the running search; returns a waypoint list once finished, else None."""
        S = self.search
        if S is None:
            return None
        if S["done"] is not None:
            return S["done"]
        t_end = time.time() + budget_s
        goal, openh, g, came = S["goal"], S["open"], S["g"], S["from"]
        while openh:
            S["n"] += 1
            if S["n"] > 30000:
                break
            if (S["n"] & 31) == 0 and time.time() > t_end:
                return None
            _, gc, cur = heapq.heappop(openh)
            if gc > g.get(cur, 1e18):
                continue
            if cur == goal:
                S["done"] = self._finish(cur, came, S["target"])
                return S["done"]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    nb = (cur[0] + di, cur[1] + dj)
                    if not self.walkable(*self._center(nb)):
                        continue
                    if di and dj and not (self.walkable(*self._center((cur[0] + di, cur[1])))
                                          and self.walkable(*self._center((cur[0], cur[1] + dj)))):
                        continue  # no corner cutting
                    ng = gc + (1.4142 if di and dj else 1.0)
                    if ng < g.get(nb, 1e18):
                        g[nb] = ng
                        came[nb] = cur
                        dx, dy = abs(goal[0] - nb[0]), abs(goal[1] - nb[1])
                        hh = (dx + dy) + (0.4142 - 1.0) * min(dx, dy)
                        heapq.heappush(openh, (ng + hh, ng, nb))
        S["done"] = []  # unreachable or too big: empty path means "steer directly"
        return S["done"]

    def _finish(self, cur, came, target):
        cells = [cur]
        while cur in came:
            cur = came[cur]
            cells.append(cur)
        cells.reverse()
        pts = [self._center(c) for c in cells]
        pts.append((target[0], target[1]))
        # string-pulling: drop waypoints that a straight, wall-clear segment already covers
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._seg_clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _seg_clear(self, a, b):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / 2.0))
        for k in range(1, n + 1):
            x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
            if not self.walkable(x, y):
                return False
        return True
