import * as ort from 'onnxruntime-web/wasm';
import type { CapturedFrame, CaptureQuality } from './types';

/** Callbacks wired by the owning screen. Mirroring stays CSS-only on the preview. */
export interface StilltypeCameraCallbacks {
  onQuality: (quality: CaptureQuality) => void;
  onError: (message: string) => void;
}

export type FramesHandler = (frames: CapturedFrame[]) => Promise<void>;
export type PauseHandler = () => void;

/**
 * Face detector: OpenCV Zoo YuNet dynamic-input artifact, 2026may release
 * (https://huggingface.co/pollen-robotics/face_detection_yunet_2026may).
 * Served locally at MODEL_PATH and run in-process via onnxruntime-web WASM.
 *
 * Honest provenance note: the model card states the learned weights are
 * identical to the fixed-shape 2023mar variant — 2026may is the
 * dynamic-input-shape redistribution of the same YuNet architecture, not a new
 * 2026 research architecture. Recognition itself (the actual 2026 USR2 model)
 * runs in the engine and is out of scope for this module.
 */
const MODEL_PATH = '/vision/face_detection_yunet_2026may.onnx';
const ORT_WASM_PATH = '/vision/ort/';

const ENCODE_WIDTH = 640;
const ENCODE_HEIGHT = 480;
const JPEG_QUALITY = 0.75;

/** Detector input. Both dims are multiples of 32 (largest YuNet stride). */
const DET_W = 320;
const DET_H = 256;
const DET_STRIDES = [8, 16, 32];
const FACE_SCORE_THRESHOLD = 0.75;

/** Detection-pixel to encoded-pixel scale (direct stretch of 640x480 to 320x256). */
const SCALE_X = ENCODE_WIDTH / DET_W;
const SCALE_Y = ENCODE_HEIGHT / DET_H;

/** Detection runs at most this fast; capture batches flush at the same cadence. */
const DETECT_INTERVAL_MS = 1000 / 25;
const QUALITY_INTERVAL_MS = 200;
const BATCH_INTERVAL_MS = 200;
/** Bounded backpressure: at most this many unflushed batches wait on slow writers. */
const MAX_QUEUED_BATCHES = 3;

/** Mouth must move this long before a following stillness counts as a pause. */
const SPEECH_MIN_MS = 1500;
/** Stillness shorter than this resets a sub-threshold speech blip. */
const BLIP_RESET_MS = 400;
/** Hysteresis on the smoothed mouth-patch difference signal. */
const MOTION_START = 0.09;
const MOTION_STOP = 0.045;
/** Mouth patch resampled to this fixed grid so frames stay comparable. */
const PATCH_W = 24;
const PATCH_H = 18;
const PATCH_SMOOTH = 0.35;

const DARK_THRESHOLD = 0.25;
const RESTLESS_THRESHOLD = 0.7;

interface Point {
  x: number;
  y: number;
}

/** Best decoded face: capture-space landmarks plus detection-space speech cues. */
interface DetFace {
  landmarks: [number, number][];
  mouth: Point;
  eyeDist: number;
}

function friendlyOpenError(name: string, detail: string): string {
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was denied. Allow camera access in the browser or system settings, then try again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No camera was found. Connect a camera and try again.';
    case 'NotReadableError':
      return 'The camera is already in use by another app. Close the other app and try again.';
    case 'AbortError':
      return 'The camera request was cancelled. Try again.';
    default:
      return detail ? `Could not open the camera (${name}): ${detail}` : 'Could not open the camera. Try again.';
  }
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * Browser camera plus YuNet face-landmark capture.
 *
 * Contract notes for the integrator:
 * - `open()` rejects with an Error on failure (permission denied, no device,
 *   detector assets missing). Wire `try/catch` around it. `onError` is reserved
 *   for runtime failures while the camera is open (persistent detection
 *   failure, frame-writer backpressure/failure).
 * - The preview `<video>` is never transformed here. Apply `scaleX(-1)` in CSS
 *   when the mirrored setting is on; captured pixels and landmarks are always
 *   unmirrored image coordinates.
 * - `onPause` is only an indication that speech stopped; phrase segmentation
 *   stays with the caller. No inference beyond face detection runs here.
 * - Frames never touch the network or localStorage. `close()` stops the
 *   camera tracks and releases the detector session; unmount paths must call it.
 */
export class StilltypeCamera {
  public isOpen = false;

  private readonly video: HTMLVideoElement;
  private readonly callbacks: StilltypeCameraCallbacks;

  private stream: MediaStream | null = null;
  private session: ort.InferenceSession | null = null;
  private sessionGen = 0;
  private inferenceInFlight = false;
  private timerId = 0;

  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private detCanvas: HTMLCanvasElement | null = null;
  private detCtx: CanvasRenderingContext2D | null = null;
  private inputData: Float32Array | null = null;

  private recording = false;
  private onFrames: FramesHandler | null = null;
  private onPause: PauseHandler | null = null;
  private pauseMs = 700;
  private startTimestamp = 0;

  private batch: CapturedFrame[] = [];
  private lastFlushMs = 0;
  private writeChain: Promise<void> = Promise.resolve();
  private queuedBatches = 0;

  private lastDetectMs = -1;
  private lastCompleteMs = 0;
  private lastQualityMs = 0;

  private faceDetected = false;
  private brightness = 0;
  private motion = 0;
  private fps = 0;
  private fpsIntervalEma = 0;

  private prevPatch: Float32Array | null = null;
  private patchDiffEma = 0;
  private lastSpeechMs = 0;
  private speaking = false;
  private speakMs = 0;
  private stillMs = 0;
  private pauseFired = false;

  private consecutiveDetectErrors = 0;
  private detectErrorReported = false;

  constructor(video: HTMLVideoElement, callbacks: StilltypeCameraCallbacks) {
    this.video = video;
    this.callbacks = callbacks;
  }

  /** Devices available without opening the camera. Labels need prior permission. */
  static async getDevices(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  /**
   * Opens the camera, loads the YuNet detector and starts the preview loop.
   * Rejects with a human-readable Error; safe to call again after close().
   * Returns the video-input device list (labels populated once permitted).
   */
  async open(cameraId?: string): Promise<MediaDeviceInfo[]> {
    if (this.isOpen) this.close();
    this.sessionGen += 1;

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 25 },
          deviceId: cameraId ? { exact: cameraId } : undefined,
        },
        audio: false,
      });
    } catch (error) {
      const name = error instanceof DOMException ? error.name : 'Error';
      const detail = error instanceof Error ? error.message : '';
      throw new Error(friendlyOpenError(name, detail));
    }
    this.stream = stream;

    const video = this.video;
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;

    try {
      await this.waitForMetadata(video);
      await video.play();
    } catch (error) {
      this.stopTracks();
      this.stream = null;
      const detail = error instanceof Error ? error.message : '';
      throw new Error(detail || 'Could not start the camera preview. Try again.');
    }

    try {
      // Single-threaded WASM: no cross-origin isolation needed (Tauri-safe).
      // Uses the wasm-only onnxruntime-web bundle, which requests exactly the
      // ort-wasm-simd-threaded.mjs/.wasm files served under ORT_WASM_PATH.
      ort.env.wasm.wasmPaths = ORT_WASM_PATH;
      ort.env.wasm.numThreads = 1;
      this.session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['wasm'],
      });
    } catch (error) {
      this.stopTracks();
      this.stream = null;
      const detail = error instanceof Error ? error.message : '';
      throw new Error(
        detail || 'The face detector failed to load. Check that the app assets are installed, then try again.',
      );
    }

    this.canvas = document.createElement('canvas');
    this.canvas.width = ENCODE_WIDTH;
    this.canvas.height = ENCODE_HEIGHT;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.detCanvas = document.createElement('canvas');
    this.detCanvas.width = DET_W;
    this.detCanvas.height = DET_H;
    this.detCtx = this.detCanvas.getContext('2d', { willReadFrequently: true });
    this.inputData = new Float32Array(3 * DET_H * DET_W);

    this.isOpen = true;
    this.resetRuntimeState();
    this.startLoop();

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((device) => device.kind === 'videoinput');
    } catch {
      return [];
    }
  }

  /**
   * Starts recording with a fresh clock. Batches of roughly 5 frames flush
   * about every 200 ms; writer calls are serialized with bounded backpressure.
   * Throws when no camera is open or a recording is already running.
   */
  start(onFrames: FramesHandler, onPause?: PauseHandler, pauseSeconds = 0.7): void {
    if (!this.isOpen || !this.session || !this.stream) {
      throw new Error('Open the camera before recording.');
    }
    if (this.recording) {
      throw new Error('Recording is already in progress.');
    }
    this.onFrames = onFrames;
    this.onPause = onPause ?? null;
    this.pauseMs = Math.max(0.2, pauseSeconds) * 1000;
    this.startTimestamp = performance.now();
    this.batch = [];
    this.lastFlushMs = performance.now();
    this.writeChain = Promise.resolve();
    this.queuedBatches = 0;
    this.speaking = false;
    this.speakMs = 0;
    this.stillMs = 0;
    this.pauseFired = false;
    this.prevPatch = null;
    this.patchDiffEma = 0;
    this.lastSpeechMs = performance.now();
    this.recording = true;
  }

  /** Stops recording, flushes remaining frames and awaits queued writes. Preview continues. */
  async stop(): Promise<void> {
    if (!this.recording) return;
    this.recording = false;
    if (this.batch.length > 0) {
      this.enqueueBatch(this.batch);
      this.batch = [];
    }
    this.onFrames = null;
    await this.writeChain;
  }

  /**
   * Releases everything: stops recording state (unflushed frames are
   * discarded), stops camera tracks, cancels the loop, releases the detector
   * session, clears buffers and detaches the video element.
   */
  close(): void {
    this.sessionGen += 1;
    this.recording = false;
    this.batch = [];
    this.queuedBatches = 0;
    this.onFrames = null;
    this.onPause = null;
    this.isOpen = false;
    this.inferenceInFlight = false;

    // Bounded timer cadence, not rAF: capture continues while the main
    // window is hidden behind the overlay. clearTimeout cancels the loop.
    if (this.timerId !== 0) {
      window.clearTimeout(this.timerId);
      this.timerId = 0;
    }
    this.stopTracks();
    this.video.removeAttribute('src');
    this.video.srcObject = null;

    const session = this.session;
    this.session = null;
    if (session) {
      // Release is async; a stale in-flight run discards itself via sessionGen.
      void session.release().catch(() => {
        // Release failure after teardown must not break close().
      });
    }
    this.canvas = null;
    this.ctx = null;
    this.detCanvas = null;
    this.detCtx = null;
    this.inputData = null;
    this.prevPatch = null;
    this.resetRuntimeState();
  }

  private resetRuntimeState(): void {
    this.faceDetected = false;
    this.brightness = 0;
    this.motion = 0;
    this.fps = 0;
    this.fpsIntervalEma = 0;
    this.prevPatch = null;
    this.patchDiffEma = 0;
    this.lastSpeechMs = 0;
    this.speaking = false;
    this.speakMs = 0;
    this.stillMs = 0;
    this.pauseFired = false;
    this.lastDetectMs = -1;
    this.lastCompleteMs = 0;
    this.lastQualityMs = 0;
    this.consecutiveDetectErrors = 0;
    this.detectErrorReported = false;
  }

  private stopTracks(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
  }

  private waitForMetadata(video: HTMLVideoElement): Promise<void> {
    if (video.readyState >= 1 && video.videoWidth > 0) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        cleanup();
        reject(new Error('The camera did not respond in time. Try again.'));
      }, 10000);
      const cleanup = () => {
        window.clearTimeout(timer);
        video.removeEventListener('loadedmetadata', onMetadata);
        video.removeEventListener('error', onFailure);
      };
      const onMetadata = () => {
        cleanup();
        resolve();
      };
      const onFailure = () => {
        cleanup();
        reject(new Error('Could not start the camera preview. Try again.'));
      };
      video.addEventListener('loadedmetadata', onMetadata, { once: true });
      video.addEventListener('error', onFailure, { once: true });
    });
  }

  private startLoop(): void {
    // Timer cadence, never rAF: the detector must keep running while the
    // main window is hidden and the overlay is showing. At most one
    // inference is ever in flight; close() clears the pending timeout.
    if (this.timerId !== 0) return; // never run duplicate loops after reopen
    const tick = () => {
      if (!this.isOpen) {
        this.timerId = 0;
        return;
      }
      this.tick(performance.now());
      if (!this.isOpen) {
        this.timerId = 0;
        return;
      }
      this.timerId = window.setTimeout(tick, DETECT_INTERVAL_MS);
    };
    this.timerId = window.setTimeout(tick, 0);
  }

  private tick(now: number): void {
    if (now - this.lastDetectMs < DETECT_INTERVAL_MS) return;
    if (!this.session || this.inferenceInFlight) return;
    if (this.video.readyState < 2 || this.video.videoWidth === 0) return;
    this.lastDetectMs = now;
    this.drawFrame();
    // Async inference: at most one run at a time, never overlapping.
    // A stale completion (after close/reopen) discards itself via sessionGen.
    void this.detect(now);
  }

  private async detect(now: number): Promise<void> {
    const session = this.session;
    const canvas = this.canvas;
    if (!session || !this.detCtx || !this.detCanvas || !this.inputData || !canvas) return;
    const gen = this.sessionGen;
    this.inferenceInFlight = true;

    this.detCtx.drawImage(canvas, 0, 0, DET_W, DET_H);
    const image = this.detCtx.getImageData(0, 0, DET_W, DET_H);
    this.buildInput(image);
    const input = new ort.Tensor('float32', this.inputData, [1, 3, DET_H, DET_W]);

    let feeds: ort.InferenceSession.ReturnType | null = null;
    try {
      feeds = await session.run({ input });
      if (gen !== this.sessionGen || !this.isOpen) return;
      const det = this.decodeBest(feeds);
      this.faceDetected = det !== null;
      this.consecutiveDetectErrors = 0;
      this.updateFps(now);
      this.updateMotion(now, det, image);

      if (now - this.lastQualityMs >= QUALITY_INTERVAL_MS) {
        this.lastQualityMs = now;
        this.brightness = this.measureBrightness();
        this.callbacks.onQuality({
          faceDetected: this.faceDetected,
          brightness: this.brightness,
          motion: this.motion,
          fps: Math.round(this.fps * 10) / 10,
          hint: this.pickHint(),
        });
      }

      if (!this.recording) return;
      // Frames without a face still record (null landmarks) so the clip keeps
      // its cadence; the engine decides usability. A failed decode never
      // fabricates points.
      this.batch.push({
        jpeg: this.encodeJpeg(),
        timestamp: Math.round(now - this.startTimestamp),
        landmarks: det ? det.landmarks : null,
      });
      if (now - this.lastFlushMs >= BATCH_INTERVAL_MS && this.batch.length > 0) {
        this.lastFlushMs = now;
        this.enqueueBatch(this.batch);
        this.batch = [];
      }
    } catch {
      // A detection error is never reported as a valid face.
      if (gen !== this.sessionGen || !this.isOpen) return;
      this.faceDetected = false;
      this.consecutiveDetectErrors += 1;
      if (this.consecutiveDetectErrors >= 30 && !this.detectErrorReported) {
        this.detectErrorReported = true;
        this.callbacks.onError('Face detection keeps failing. Reopen the camera to try again.');
      }
    } finally {
      if (feeds) {
        // Release runtime-allocated output tensors. The input tensor wraps
        // our reused buffer and is left for GC; the buffer itself is retained.
        for (const name of Object.keys(feeds)) {
          const tensor = feeds[name] as unknown as { dispose?: unknown };
          if (typeof tensor?.dispose === 'function') tensor.dispose();
        }
      }
      if (gen === this.sessionGen) this.inferenceInFlight = false;
    }
  }

  /** Unmirrored contain-fit draw of the live video into the 640x480 buffer. */
  private drawFrame(): void {
    if (!this.ctx || !this.canvas) return;
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (vw === 0 || vh === 0) return;
    const scale = Math.min(ENCODE_WIDTH / vw, ENCODE_HEIGHT / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const dx = (ENCODE_WIDTH - dw) / 2;
    const dy = (ENCODE_HEIGHT - dh) / 2;
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, ENCODE_WIDTH, ENCODE_HEIGHT);
    this.ctx.drawImage(this.video, dx, dy, dw, dh);
  }

  /** YuNet input: BGR planar float32, raw 0..255, no normalization. */
  private buildInput(image: ImageData): void {
    if (!this.inputData) return;
    const { data } = image;
    const area = DET_W * DET_H;
    const out = this.inputData;
    for (let p = 0; p < area; p += 1) {
      const o = p * 4;
      out[p] = data[o + 2];
      out[area + p] = data[o + 1];
      out[area * 2 + p] = data[o];
    }
  }

  private outputData(
    feeds: ort.InferenceSession.ReturnType,
    name: string,
    expected: number,
  ): Float32Array | null {
    const value = feeds[name] as unknown as ort.Tensor | undefined;
    const data = value ? (value.data as unknown) : null;
    if (!(data instanceof Float32Array) || data.length < expected) return null;
    return data;
  }

  /**
   * Decodes the 12 YuNet heads (strides 8/16/32, one anchor per cell) and
   * returns the highest-scoring valid face above threshold, or null when no
   * face is present. No NMS: a single best face is all this module needs.
   * Per the upstream model card, kps order is right eye, left eye, nose,
   * right/left mouth corner. Output order is always image-left eye,
   * image-right eye, nose base, mouth center (mouth-corner midpoint).
   */
  private decodeBest(feeds: ort.InferenceSession.ReturnType): DetFace | null {
    let best: { score: number; re: Point; le: Point; nose: Point; mr: Point; ml: Point } | null =
      null;
    for (const s of DET_STRIDES) {
      const w = DET_W / s;
      const h = DET_H / s;
      const count = w * h;
      const cls = this.outputData(feeds, `cls_${s}`, count);
      const obj = this.outputData(feeds, `obj_${s}`, count);
      const bbox = this.outputData(feeds, `bbox_${s}`, count * 4);
      const kps = this.outputData(feeds, `kps_${s}`, count * 10);
      if (!cls || !obj || !bbox || !kps) continue;
      for (let i = 0; i < count; i += 1) {
        const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
        if (!(score >= FACE_SCORE_THRESHOLD)) continue;
        if (best && score <= best.score) continue;
        const row = Math.floor(i / w);
        const col = i % w;
        const b = i * 4;
        const bw = Math.exp(bbox[b + 2]) * s;
        const bh = Math.exp(bbox[b + 3]) * s;
        if (!Number.isFinite(bw) || !Number.isFinite(bh) || bw <= 0 || bh <= 0) continue;
        const k = i * 10;
        const pts: Point[] = [];
        let valid = true;
        for (let j = 0; j < 5; j += 1) {
          const x = (col + kps[k + j * 2]) * s;
          const y = (row + kps[k + j * 2 + 1]) * s;
          // Small margin: a face may touch the frame edge; garbage may not.
          if (!Number.isFinite(x) || !Number.isFinite(y) || x < -4 || y < -4 || x > DET_W + 4 || y > DET_H + 4) {
            valid = false;
            break;
          }
          pts.push({ x, y });
        }
        if (!valid) continue;
        best = { score, re: pts[0], le: pts[1], nose: pts[2], mr: pts[3], ml: pts[4] };
      }
    }
    if (!best) return null;
    const eyeDist = Math.hypot(best.re.x - best.le.x, best.re.y - best.le.y);
    if (!(eyeDist > 4)) return null; // reject degenerate tiny detections
    const round1 = (v: number): number => Math.round(v * 10) / 10;
    const toCapture = (p: Point): [number, number] => [round1(p.x * SCALE_X), round1(p.y * SCALE_Y)];
    const eyeA = toCapture(best.re);
    const eyeB = toCapture(best.le);
    const eyes: [[number, number], [number, number]] =
      eyeA[0] <= eyeB[0] ? [eyeA, eyeB] : [eyeB, eyeA];
    const mouthDet: Point = { x: (best.mr.x + best.ml.x) / 2, y: (best.mr.y + best.ml.y) / 2 };
    return { landmarks: [eyes[0], eyes[1], toCapture(best.nose), toCapture(mouthDet)], mouth: mouthDet, eyeDist };
  }

  private updateFps(now: number): void {
    if (this.lastCompleteMs > 0) {
      const interval = now - this.lastCompleteMs;
      if (interval > 0) {
        this.fpsIntervalEma = this.fpsIntervalEma * 0.9 + interval * 0.1;
      }
    } else {
      this.fpsIntervalEma = DETECT_INTERVAL_MS;
    }
    this.lastCompleteMs = now;
    this.fps = 1000 / this.fpsIntervalEma;
  }

  /**
   * Mouth motion from brightness-normalized image difference on a small patch
   * around the decoded mouth center, scaled by eye distance. The patch is
   * resampled to a fixed grid and divided by its mean, so global light changes
   * do not read as motion. Speech stillness only advances while a face is
   * visible, so losing the face can never fake a pause. `onPause` fires at
   * most once per speech segment and only while recording; it is an
   * indication, and phrase segmentation stays with the caller.
   */
  private updateMotion(now: number, det: DetFace | null, image: ImageData | null): void {
    const dtMs = Math.min(1000, Math.max(0, now - this.lastSpeechMs));
    this.lastSpeechMs = now;
    if (!det || !image) {
      // Freeze timers and drop the reference patch: a reacquired face starts
      // clean instead of spiking against a stale patch.
      this.prevPatch = null;
      return;
    }
    const half = Math.min(52, Math.max(10, det.eyeDist * 0.55));
    const patch = this.sampleMouthPatch(image, det.mouth.x, det.mouth.y, half, half * 0.75);
    const prev = this.prevPatch;
    this.prevPatch = patch;
    if (!prev) return;
    let diff = 0;
    for (let i = 0; i < patch.length; i += 1) diff += Math.abs(patch[i] - prev[i]);
    diff /= patch.length;
    this.patchDiffEma = this.patchDiffEma * (1 - PATCH_SMOOTH) + diff * PATCH_SMOOTH;
    this.motion = clamp01((this.patchDiffEma - 0.02) / 0.25);

    if (this.speaking) {
      if (this.patchDiffEma < MOTION_STOP) this.speaking = false;
    } else if (this.patchDiffEma > MOTION_START) {
      this.speaking = true;
      if (this.pauseFired || this.stillMs > BLIP_RESET_MS) {
        // New speech segment after a pause or a settled stillness.
        this.speakMs = 0;
        this.pauseFired = false;
      }
      this.stillMs = 0;
    }

    if (this.speaking) {
      this.speakMs += dtMs;
      this.stillMs = 0;
    } else {
      this.stillMs += dtMs;
      if (this.stillMs > BLIP_RESET_MS && this.speakMs < SPEECH_MIN_MS) {
        // Short blip that never became speech; forget it.
        this.speakMs = 0;
      }
      if (
        this.recording &&
        this.onPause &&
        !this.pauseFired &&
        this.speakMs >= SPEECH_MIN_MS &&
        this.stillMs >= this.pauseMs
      ) {
        this.pauseFired = true;
        try {
          this.onPause();
        } catch {
          // A throwing pause handler must not break the capture loop.
        }
      }
    }
  }

  private sampleMouthPatch(
    image: ImageData,
    cx: number,
    cy: number,
    halfW: number,
    halfH: number,
  ): Float32Array {
    const out = new Float32Array(PATCH_W * PATCH_H);
    const { data, width, height } = image;
    let sum = 0;
    for (let gy = 0; gy < PATCH_H; gy += 1) {
      const sy = Math.min(
        height - 1,
        Math.max(0, Math.round(cy - halfH + ((gy + 0.5) * 2 * halfH) / PATCH_H)),
      );
      for (let gx = 0; gx < PATCH_W; gx += 1) {
        const sx = Math.min(
          width - 1,
          Math.max(0, Math.round(cx - halfW + ((gx + 0.5) * 2 * halfW) / PATCH_W)),
        );
        const o = (sy * width + sx) * 4;
        const gray = 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2];
        out[gy * PATCH_W + gx] = gray;
        sum += gray;
      }
    }
    const mean = sum / out.length;
    if (mean > 1) {
      for (let i = 0; i < out.length; i += 1) out[i] /= mean;
    }
    return out;
  }

  private encodeJpeg(): string {
    if (!this.canvas) return '';
    const url = this.canvas.toDataURL('image/jpeg', JPEG_QUALITY);
    const comma = url.indexOf(',');
    return comma >= 0 ? url.slice(comma + 1) : '';
  }

  private measureBrightness(): number {
    if (!this.ctx) return 0;
    try {
      const data = this.ctx.getImageData(0, 0, ENCODE_WIDTH, ENCODE_HEIGHT).data;
      let sum = 0;
      let count = 0;
      // Sample every 16th pixel; plenty for a 0..1 ambient-light estimate.
      for (let i = 0; i < data.length; i += 64) {
        sum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        count += 1;
      }
      return count === 0 ? 0 : clamp01(sum / count / 255);
    } catch {
      return 0;
    }
  }

  private enqueueBatch(frames: CapturedFrame[]): void {
    const handler = this.onFrames;
    if (!handler || frames.length === 0) return;
    if (this.queuedBatches >= MAX_QUEUED_BATCHES) {
      this.failRecording('Recording stopped: the frame writer fell behind. Try again.');
      return;
    }
    const snapshot = frames.slice();
    this.queuedBatches += 1;
    this.writeChain = this.writeChain.then(async () => {
      try {
        await handler(snapshot);
      } catch {
        this.failRecording('Recording stopped: saving frames failed. Try again.');
      } finally {
        this.queuedBatches = Math.max(0, this.queuedBatches - 1);
      }
    });
  }

  /** Stops recording on error and drops the pending batch; preview continues. */
  private failRecording(message: string): void {
    if (this.recording) {
      this.recording = false;
      this.batch = [];
      this.onFrames = null;
      this.callbacks.onError(message);
    } else if (this.isOpen) {
      // A queued write failed after stop(): report without touching state.
      this.callbacks.onError(message);
    }
  }

  private pickHint(): string {
    if (!this.faceDetected) return 'Face the camera';
    if (this.brightness < DARK_THRESHOLD) return 'More light';
    if (this.motion > RESTLESS_THRESHOLD && !this.speaking) return 'Keep your head still';
    return 'Ready to dictate';
  }
}
