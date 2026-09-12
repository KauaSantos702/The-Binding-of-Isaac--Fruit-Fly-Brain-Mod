--[[ hud.lua -- bars and graphs for watching the brain think.

  The Isaac API has no primitive for drawing a rectangle, so everything here
  is one 4x4 white sprite, tinted and scaled.  Its pivot is at (0,0), so it
  grows right and down from wherever it is rendered.
]]

local Hud = {}

local ZERO = Vector(0, 0)
local px

local function pixel()
  if not px then
    px = Sprite()
    px:Load("gfx/ui/pixel.anm2", true)
    px:SetFrame("Idle", 0)
  end
  return px
end

function Hud.rect(x, y, w, h, r, g, b, a)
  if w <= 0 or h <= 0 then return end
  local s = pixel()
  s.Scale = Vector(w / 4, h / 4)
  s.Color = Color(r, g, b, a or 1.0)
  s:Render(Vector(x, y), ZERO, ZERO)
end

function Hud.text(str, x, y, r, g, b, a)
  Isaac.RenderText(str, x, y, r or 0.75, g or 0.95, b or 0.9, a or 1.0)
end

-- Small text, the size the game's own HUD counters use.
function Hud.stext(str, x, y, scale, col, a)
  col = col or { 0.8, 0.85, 0.9 }
  Isaac.RenderScaledText(str, x, y, scale or 0.5, scale or 0.5,
    col[1], col[2], col[3], a or 1.0)
end

---------------------------------------------------------------------------
-- thin bars, for the compact readout
---------------------------------------------------------------------------

-- A slim filled bar with a dark track and a soft edge highlight.
function Hud.thinBar(x, y, w, h, frac, col, a)
  if frac < 0 then frac = 0 elseif frac > 1 then frac = 1 end
  a = a or 1.0
  Hud.rect(x, y, w, h, 0.06, 0.06, 0.09, 0.55 * a)
  local fw = w * frac
  if fw >= 1 then
    Hud.rect(x, y, fw, h, col[1], col[2], col[3], 0.9 * a)
    Hud.rect(x, y, fw, 1, col[1] * 0.4 + 0.6, col[2] * 0.4 + 0.6,
      col[3] * 0.4 + 0.6, 0.55 * a)
  end
end

-- A slim bar that grows out from the centre, for signed quantities.
function Hud.thinBiBar(x, y, w, h, value, colL, colR, a)
  if value < -1 then value = -1 elseif value > 1 then value = 1 end
  a = a or 1.0
  Hud.rect(x, y, w, h, 0.06, 0.06, 0.09, 0.55 * a)
  local mid = x + w * 0.5
  local half = w * 0.5 * value
  if half < 0 then
    Hud.rect(mid + half, y, -half, h, colL[1], colL[2], colL[3], 0.9 * a)
  elseif half > 0 then
    Hud.rect(mid, y, half, h, colR[1], colR[2], colR[3], 0.9 * a)
  end
  Hud.rect(mid, y - 1, 1, h + 2, 0.85, 0.88, 0.92, 0.5 * a)
end

---------------------------------------------------------------------------
-- a labelled 0..100 bar
---------------------------------------------------------------------------
-- `frac` is clamped to 0..1.  `col` is {r, g, b}.
function Hud.bar(x, y, w, h, frac, label, col, valueText)
  if frac < 0 then frac = 0 elseif frac > 1 then frac = 1 end
  Hud.rect(x, y, w, h, 0.10, 0.10, 0.14, 0.85)
  Hud.rect(x, y, w * frac, h, col[1], col[2], col[3], 0.95)
  -- a tick every 25%
  for k = 1, 3 do
    Hud.rect(x + w * k * 0.25, y, 1, h, 0, 0, 0, 0.45)
  end
  Hud.text(label, x - 74, y - 2, 0.65, 0.8, 0.85)
  Hud.text(valueText or string.format("%d", frac * 100 + 0.5), x + w + 6, y - 2,
    col[1], col[2], col[3])
end

-- a bar that runs out from the centre, for signed quantities like turning
function Hud.biBar(x, y, w, h, value, label, colL, colR, valueText)
  if value < -1 then value = -1 elseif value > 1 then value = 1 end
  Hud.rect(x, y, w, h, 0.10, 0.10, 0.14, 0.85)
  local mid = x + w * 0.5
  if value < 0 then
    Hud.rect(mid + w * 0.5 * value, y, -w * 0.5 * value, h, colL[1], colL[2], colL[3], 0.95)
  else
    Hud.rect(mid, y, w * 0.5 * value, h, colR[1], colR[2], colR[3], 0.95)
  end
  Hud.rect(mid, y - 1, 1, h + 2, 0.9, 0.9, 0.9, 0.7)
  Hud.text(label, x - 74, y - 2, 0.65, 0.8, 0.85)
  if valueText then Hud.text(valueText, x + w + 6, y - 2, 0.8, 0.8, 0.85) end
end

---------------------------------------------------------------------------
-- a scrolling column graph over a ring buffer
---------------------------------------------------------------------------
local History = {}
History.__index = History

function History.new(size)
  return setmetatable({ size = size, n = 0, head = 0, v = {} }, History)
end

function History:push(x)
  self.head = self.head % self.size + 1
  self.v[self.head] = x
  if self.n < self.size then self.n = self.n + 1 end
end

-- oldest first
function History:each(fn)
  local start = self.head - self.n
  for k = 1, self.n do
    local i = (start + k - 1) % self.size + 1
    fn(k, self.v[i])
  end
end

Hud.History = History

function Hud.graph(x, y, w, h, hist, maxValue, col, label)
  Hud.rect(x, y, w, h, 0.08, 0.08, 0.11, 0.80)
  Hud.rect(x, y + h, w, 1, 0.4, 0.4, 0.45, 0.6)
  if maxValue <= 0 then maxValue = 1 end
  local cw = w / hist.size
  if cw < 1 then cw = 1 end
  hist:each(function(k, v)
    if not v or v <= 0 then return end
    local f = v / maxValue
    if f > 1 then f = 1 end
    local ch = h * f
    Hud.rect(x + (k - 1) * cw, y + h - ch, cw, ch, col[1], col[2], col[3], 0.9)
  end)
  if label then Hud.text(label, x, y - 11, col[1], col[2], col[3]) end
end

---------------------------------------------------------------------------
-- a row of cells, one per neuron population
---------------------------------------------------------------------------
function Hud.popRow(x, y, cellW, cellH, entries, maxHz)
  for k = 1, #entries do
    local e = entries[k]
    local f = e.hz / maxHz
    if f > 1 then f = 1 end
    local cx = x + (k - 1) * (cellW + 2)
    Hud.rect(cx, y, cellW, cellH, 0.10, 0.10, 0.14, 0.85)
    Hud.rect(cx, y + cellH * (1 - f), cellW, cellH * f,
      0.25 + 0.75 * f, 0.95 - 0.55 * f, 1.0 - 0.85 * f, 0.95)
    Hud.text(e.name, cx - 2, y + cellH + 2, 0.6, 0.75, 0.8)
  end
end

return Hud
