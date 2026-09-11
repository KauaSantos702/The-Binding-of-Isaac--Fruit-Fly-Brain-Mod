--[[ loader.lua -- decode the packed connectome into flat Lua arrays.

  The CSR arrays are shipped base64-encoded inside net_csr.lua because Isaac's
  Lua sandbox has no file I/O and a 116k-element table constructor would blow
  past the parser's constant limit.  Decoding is done with string.byte and
  string.unpack in large chunks, so the whole 84k-edge network loads in one
  pass with only a few hundred C calls rather than a few hundred thousand.
]]

local Loader = {}

local byte, char, sub, rep = string.byte, string.char, string.sub, string.rep
local unpack_ = string.unpack
local concat = table.concat

---------------------------------------------------------------------------
-- base64
---------------------------------------------------------------------------
local B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
local DEC = {}
for i = 1, 64 do DEC[byte(B64, i)] = i - 1 end
local PAD = byte("=")

-- Accepts either a string or the array of chunks net_csr.lua ships, then
-- decodes 4 characters to 3 bytes.
function Loader.b64decode(s)
  if type(s) == "table" then s = concat(s) end
  if type(s) ~= "string" then
    error("b64decode expected a string or chunk array, got " .. type(s), 2)
  end
  s = s:gsub("[^%w%+/=]", "")
  local n = #s
  local out, on = {}, 0
  local i = 1
  -- process in blocks so the intermediate string table stays small
  while i <= n do
    local stop = i + 4095
    if stop > n then stop = n end
    stop = stop - ((stop - i + 1) % 4)          -- keep blocks 4-char aligned
    if stop < i then stop = n end
    local piece = {}
    local pn = 0
    local j = i
    while j <= stop do
      local c1, c2, c3, c4 = byte(s, j, j + 3)
      local a, b = DEC[c1], DEC[c2]
      local v = a * 262144 + b * 4096
      if c3 == PAD then
        pn = pn + 1; piece[pn] = char(v // 65536)
      elseif c4 == PAD then
        v = v + DEC[c3] * 64
        pn = pn + 1; piece[pn] = char(v // 65536, (v // 256) % 256)
      else
        v = v + DEC[c3] * 64 + DEC[c4]
        pn = pn + 1; piece[pn] = char(v // 65536, (v // 256) % 256, v % 256)
      end
      j = j + 4
    end
    on = on + 1
    out[on] = concat(piece)
    i = stop + 1
  end
  return concat(out)
end

---------------------------------------------------------------------------
-- bulk numeric unpack
---------------------------------------------------------------------------
local CHUNK = 256

-- fmt is a single string.unpack option: "I4", "I2", "i2", ...
local function unpackArray(s, fmt, count)
  local out = {}
  local wide = "<" .. rep(fmt, CHUNK)
  local one = "<" .. fmt
  local pos, n = 1, 0
  local full = count - (count % CHUNK)
  while n < full do
    local v = { unpack_(wide, s, pos) }
    pos = v[CHUNK + 1]
    for k = 1, CHUNK do out[n + k] = v[k] end
    n = n + CHUNK
  end
  while n < count do
    local val
    val, pos = unpack_(one, s, pos)
    n = n + 1
    out[n] = val
  end
  return out
end

Loader.unpackArray = unpackArray

---------------------------------------------------------------------------
-- network
---------------------------------------------------------------------------

-- Pull one field's part files in and glue their base64 back together.
-- `loadPart` is the host's module loader: include() in Isaac, require() in the
-- offline harnesses.  Each part file also assigns itself into the FlyBrainPart
-- global, which is used as a fallback if the loader drops return values.
local function gather(packed, field, loadPart)
  local names = packed.parts and packed.parts[field]
  if not names then
    error("net_csr has no part list for " .. field)
  end
  local pieces = {}
  for k = 1, #names do
    local name = names[k]
    local part = loadPart("fly_brain." .. name)
    if type(part) ~= "table" then
      part = FlyBrainPart and FlyBrainPart[name]
    end
    if type(part) ~= "table" then
      error("part " .. name .. " failed to load (got " .. type(part) .. ")")
    end
    pieces[k] = concat(part)
  end
  return concat(pieces)
end

-- packed = the table returned by net_csr.lua
function Loader.load(packed, loadPart)
  loadPart = loadPart or require
  local n, nnz = packed.n, packed.nnz
  local indptr = unpackArray(Loader.b64decode(gather(packed, "indptr", loadPart)), "I4", n + 1)
  local indices = unpackArray(Loader.b64decode(gather(packed, "indices", loadPart)), "I2", nnz)
  local raw = unpackArray(Loader.b64decode(gather(packed, "weights", loadPart)), "i2", nnz)

  local scale = packed.wscale
  local weights = raw
  for k = 1, nnz do weights[k] = raw[k] * scale end

  return { n = n, nnz = nnz, indptr = indptr, indices = indices, weights = weights }
end

return Loader
