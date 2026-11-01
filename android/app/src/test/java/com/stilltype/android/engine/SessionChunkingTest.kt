package com.stilltype.android.engine

import com.stilltype.android.core.MAX_SESSION_FRAMES
import com.stilltype.android.core.MouthFrame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Guards the 120-second session contract: nothing past 8 s may be dropped. */
class SessionChunkingTest {
  private fun timeline(count: Int, stepMs: Long, startMs: Long = 0L): List<MouthFrame> =
    List(count) { i -> MouthFrame(ByteArray(0), startMs + i * stepMs) }

  @Test fun fullSession_retainsEveryFrameIncludingLateMarkerInOrder() {
    // 120 s at 25 fps: 3001 frames, 0..120000 ms. The last frame is the late
    // distinct marker the old takeWhile(8 s) / take(200) logic dropped.
    val frames = timeline(3001, 40L)
    assertEquals(120_000L, frames.last().timestampMs - frames.first().timestampMs)

    val chunks = partitionWindows(frames)

    val flat = chunks.flatten()
    assertEquals(frames.size, flat.size)
    assertEquals(frames.map { it.timestampMs }, flat.map { it.timestampMs })
    assertEquals(frames.last().timestampMs, flat.last().timestampMs)
    assertEquals(frames.last().timestampMs, chunks.last().last().timestampMs)
    for (window in chunks) {
      val span = window.last().timestampMs - window.first().timestampMs
      assertTrue("window span $span ms exceeds 8000 ms", span <= 8_000L)
    }
  }

  @Test fun lowFrameRate_stillPartitionsByTime() {
    // 120 s at 5 fps: 601 frames. Splitting by frame count (200/chunk) would
    // give 4 windows of ~30 s; time partitioning must give ~15-16 windows.
    val frames = timeline(601, 200L)
    val chunks = partitionWindows(frames)

    assertTrue("expected time-based split, got ${chunks.size} chunks", chunks.size > 4)
    assertEquals(frames.size, chunks.sumOf { it.size })
    assertEquals(frames.map { it.timestampMs }, chunks.flatten().map { it.timestampMs })
    for (window in chunks) {
      val span = window.last().timestampMs - window.first().timestampMs
      assertTrue("window span $span ms exceeds 8000 ms", span <= 8_000L)
    }
  }

  @Test fun sessionLimits_rejectBeyond120Seconds() {
    assertNull(sessionLimitError(MAX_SESSION_FRAMES, 120_000L))
    assertEquals("invalid-clip", sessionLimitError(MAX_SESSION_FRAMES + 1, 120_000L))
    assertEquals("invalid-clip", sessionLimitError(MAX_SESSION_FRAMES, 120_001L))
    assertEquals("invalid-clip", sessionLimitError(100, 120_001L))
  }
}
