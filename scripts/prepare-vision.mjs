import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const vision = path.join(root, 'public', 'vision');
const artifact = 'face_detection_yunet_2026may.onnx';
const digest = 'ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0';
const url = `https://huggingface.co/pollen-robotics/face_detection_yunet_2026may/resolve/2b8e922362946a0db67e861bae0f77826980effd/${artifact}`;
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

await mkdir(path.join(vision, 'ort'), { recursive: true });
for (const file of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(
    path.join(root, 'node_modules', 'onnxruntime-web', 'dist', file),
    path.join(vision, 'ort', file),
  );
}

const destination = path.join(vision, artifact);
const existing = await readFile(destination).catch((error) => {
  if (error.code !== 'ENOENT') throw error;
  return null;
});
if (!existing || sha256(existing) !== digest) {
  console.log('Downloading the pinned YuNet 2026may face-tracking artifact…');
  const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw new Error(`Face model download failed: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length > 1_000_000 || sha256(bytes) !== digest) {
    throw new Error('Face model checksum mismatch. The existing artifact was not replaced.');
  }
  await writeFile(`${destination}.part`, bytes);
  await rename(`${destination}.part`, destination);
}
console.log('Local face-tracking assets ready; no runtime CDN required.');
