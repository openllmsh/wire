/**
 * Bounded image-signature + dimension inspector. Filename/extension is not
 * evidence: live Cursor output was JPEG bytes saved as `.png`. Callers pass
 * the whole buffer (or a prefix large enough for headers); parsers never
 * scan past a small header budget except JPEG SOF, which walks markers with
 * a hard byte cap.
 */

export const IMAGE_INSPECT_MAX_BYTES = 20 * 1024 * 1024;
export const IMAGE_DIMENSION_MAX = 8_192;

export type TInspectedImageMime =
  | "image/png"
  | "image/jpeg"
  | "image/webp"
  | "image/gif";

export type TInspectedImage = {
  readonly mime: TInspectedImageMime;
  readonly width: number;
  readonly height: number;
};

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;
const JPEG_SOI0 = 0xff;
const JPEG_SOI1 = 0xd8;
const GIF87 = [0x47, 0x49, 0x46, 0x38, 0x37, 0x61] as const;
const GIF89 = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61] as const;
const RIFF = [0x52, 0x49, 0x46, 0x46] as const;
const WEBP = [0x57, 0x45, 0x42, 0x50] as const;

const u16be = (bytes: Uint8Array, offset: number): number =>
  ((bytes[offset] ?? 0) << 8) | (bytes[offset + 1] ?? 0);

const u16le = (bytes: Uint8Array, offset: number): number =>
  (bytes[offset] ?? 0) | ((bytes[offset + 1] ?? 0) << 8);

const u24le = (bytes: Uint8Array, offset: number): number =>
  (bytes[offset] ?? 0) |
  ((bytes[offset + 1] ?? 0) << 8) |
  ((bytes[offset + 2] ?? 0) << 16);

const u32be = (bytes: Uint8Array, offset: number): number =>
  ((bytes[offset] ?? 0) * 0x1000000 +
    ((bytes[offset + 1] ?? 0) << 16) +
    ((bytes[offset + 2] ?? 0) << 8) +
    (bytes[offset + 3] ?? 0)) >>>
  0;

const startsWith = (
  bytes: Uint8Array,
  sig: ReadonlyArray<number>,
  offset = 0,
): boolean => {
  if (bytes.length < offset + sig.length) return false;
  for (let i = 0; i < sig.length; i++) {
    if (bytes[offset + i] !== sig[i]) return false;
  }
  return true;
};

const validDims = (width: number, height: number): boolean =>
  Number.isInteger(width) &&
  Number.isInteger(height) &&
  width > 0 &&
  height > 0 &&
  width <= IMAGE_DIMENSION_MAX &&
  height <= IMAGE_DIMENSION_MAX;

const inspectPng = (bytes: Uint8Array): TInspectedImage | null => {
  if (!startsWith(bytes, PNG_SIG)) return null;
  // IHDR: 8 sig + 4 len + 4 type + 4 width + 4 height
  if (bytes.length < 24) return null;
  if (
    bytes[12] !== 0x49 ||
    bytes[13] !== 0x48 ||
    bytes[14] !== 0x44 ||
    bytes[15] !== 0x52
  ) {
    return null;
  }
  const width = u32be(bytes, 16);
  const height = u32be(bytes, 20);
  if (!validDims(width, height)) return null;
  return { mime: "image/png", width, height };
};

/** Walk JPEG markers until SOF0/1/2; ignore SOS payload. Cap the walk. */
const inspectJpeg = (bytes: Uint8Array): TInspectedImage | null => {
  if (bytes.length < 4) return null;
  if (bytes[0] !== JPEG_SOI0 || bytes[1] !== JPEG_SOI1) return null;
  let i = 2;
  const cap = Math.min(bytes.length, 64 * 1024);
  while (i + 9 < cap) {
    if (bytes[i] !== 0xff) {
      i += 1;
      continue;
    }
    let marker = bytes[i + 1] ?? 0;
    while (marker === 0xff && i + 2 < cap) {
      i += 1;
      marker = bytes[i + 1] ?? 0;
    }
    if (marker === 0xd8 || marker === 0xd9 || marker === 0x01) {
      i += 2;
      continue;
    }
    // RST / TEM have no length
    if (marker >= 0xd0 && marker <= 0xd7) {
      i += 2;
      continue;
    }
    if (i + 4 >= cap) return null;
    const len = u16be(bytes, i + 2);
    if (len < 2) return null;
    // SOF0 / SOF1 / SOF2
    if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2) {
      if (i + 9 >= cap) return null;
      const height = u16be(bytes, i + 5);
      const width = u16be(bytes, i + 7);
      if (!validDims(width, height)) return null;
      return { mime: "image/jpeg", width, height };
    }
    if (marker === 0xda) return null; // SOS without SOF
    i += 2 + len;
  }
  return null;
};

const inspectGif = (bytes: Uint8Array): TInspectedImage | null => {
  if (!startsWith(bytes, GIF87) && !startsWith(bytes, GIF89)) return null;
  if (bytes.length < 10) return null;
  const width = u16le(bytes, 6);
  const height = u16le(bytes, 8);
  if (!validDims(width, height)) return null;
  return { mime: "image/gif", width, height };
};

const inspectWebp = (bytes: Uint8Array): TInspectedImage | null => {
  if (!startsWith(bytes, RIFF) || !startsWith(bytes, WEBP, 8)) return null;
  if (bytes.length < 30) return null;
  const fourcc =
    String.fromCharCode(bytes[12] ?? 0) +
    String.fromCharCode(bytes[13] ?? 0) +
    String.fromCharCode(bytes[14] ?? 0) +
    String.fromCharCode(bytes[15] ?? 0);
  if (fourcc === "VP8X") {
    if (bytes.length < 30) return null;
    const width = u24le(bytes, 24) + 1;
    const height = u24le(bytes, 27) + 1;
    if (!validDims(width, height)) return null;
    return { mime: "image/webp", width, height };
  }
  if (fourcc === "VP8 ") {
    // lossy: bytes 16-19 are the "VP8 " chunk size, 20-22 the 3-byte VP8
    // frame tag, THEN the 0x9d 0x01 0x2a start code at 23-25, followed by
    // 16-bit width (26-27) / height (28-29). Verified against real
    // cwebp-encoded output — see `tests/transport/image-signature-webp.test.ts`
    // — a previous off-by-3 read the start code at 20-22 and dims at
    // 23/25, which never matched a real VP8 lossy stream.
    if (bytes.length < 30) return null;
    if (bytes[23] !== 0x9d || bytes[24] !== 0x01 || bytes[25] !== 0x2a) {
      return null;
    }
    const width = u16le(bytes, 26) & 0x3fff;
    const height = u16le(bytes, 28) & 0x3fff;
    if (!validDims(width, height)) return null;
    return { mime: "image/webp", width, height };
  }
  if (fourcc === "VP8L") {
    if (bytes.length < 25) return null;
    if (bytes[20] !== 0x2f) return null;
    const bits =
      (bytes[21] ?? 0) |
      ((bytes[22] ?? 0) << 8) |
      ((bytes[23] ?? 0) << 16) |
      ((bytes[24] ?? 0) << 24);
    const width = (bits & 0x3fff) + 1;
    const height = ((bits >> 14) & 0x3fff) + 1;
    if (!validDims(width, height)) return null;
    return { mime: "image/webp", width, height };
  }
  return null;
};

/**
 * Sniff magic + dimensions. Returns null for truncated, oversized, unknown,
 * or out-of-range dimensions. Does not throw.
 */
export const inspectImageBytes = (
  bytes: Uint8Array,
): TInspectedImage | null => {
  if (bytes.byteLength === 0 || bytes.byteLength > IMAGE_INSPECT_MAX_BYTES) {
    return null;
  }
  return (
    inspectPng(bytes) ??
    inspectJpeg(bytes) ??
    inspectGif(bytes) ??
    inspectWebp(bytes)
  );
};
