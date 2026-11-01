package com.stilltype.android.data

import com.stilltype.android.core.AppData
import com.stilltype.android.core.AppSettings
import com.stilltype.android.core.HistoryEntry
import com.stilltype.android.core.MobileModelManifest
import com.stilltype.android.core.ModelAsset
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** Guards retention pruning, CSV export safety and bundled-model validation. */
class DataFormatTest {
  private val sha = "0123456789abcdef".repeat(4)

  private fun entry(createdAt: Long, text: String = "hello world"): HistoryEntry =
    HistoryEntry(
      id = UUID.randomUUID().toString(),
      text = text,
      raw = text,
      createdAt = createdAt,
      durationMs = 1000,
      inferenceMs = 50,
    )

  private fun asset(name: String, size: Long): ModelAsset =
    ModelAsset(name = name, sizeBytes = size, sha256 = sha)

  private fun validManifest(): MobileModelManifest = MobileModelManifest(
    sourceCheckpointSha256 = BundledModelRepository.EXPECTED_SOURCE_SHA,
    files = listOf(
      asset("usr2-ctc-int8.onnx", 10_000_000),
      asset("tokens.json", 10_000),
      asset("yunet-2026may.onnx", 100_000),
      asset("mean-face.json", 100),
    ),
  )

  @Test fun retention_dropsOnlyExpiredEntries() {
    val now = System.currentTimeMillis()
    val day = 24L * 60L * 60L * 1000L
    val data = AppData(
      settings = AppSettings(retentionDays = 30),
      history = listOf(entry(now - 31 * day, "old"), entry(now - 29 * day, "recent")),
    )
    val pruned = data.pruned(now)
    assertEquals(listOf("recent"), pruned.history.map { it.text })
  }

  @Test fun csv_neutralizesFormulasAndEscapesQuotes() {
    assertEquals("\"'=SUM(1,2)\"", csvCell("=SUM(1,2)"))
    assertEquals("\"'  +SUM(1,2)\"", csvCell("  +SUM(1,2)"))
    assertEquals("\"a\"\"b\"", csvCell("a\"b"))
    val csv = buildCsv(AppData(history = listOf(entry(1_700_000_000_000L, "=SUM(1,2)"))))
    assertTrue(
      csv.startsWith("id,createdAt,text,raw,favorite,words,durationMs,inferenceMs\r\n"),
    )
    assertTrue(csv.contains("\"'=SUM(1,2)\""))
  }

  @Test fun settings_accepts120AndKeepsOlderShorterDurations() {
    validateSettings(AppSettings(maxClipSeconds = 3))
    validateSettings(AppSettings(maxClipSeconds = 8))
    validateSettings(AppSettings(maxClipSeconds = 120))
    validateSettings(AppSettings())
    try {
      validateSettings(AppSettings(maxClipSeconds = 2))
      fail("expected maxClipSeconds 2 to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: below the 3-second minimum.
    }
    try {
      validateSettings(AppSettings(maxClipSeconds = 121))
      fail("expected maxClipSeconds 121 to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: above the 120-second session limit.
    }
  }

  @Test fun history_acceptsDurationThrough120Seconds() {
    validateEntry(entry(1_700_000_000_000L).copy(durationMs = 60_000))
    validateEntry(entry(1_700_000_000_000L).copy(durationMs = 120_000))
    try {
      validateEntry(entry(1_700_000_000_000L).copy(durationMs = 120_001))
      fail("expected duration beyond 120 seconds to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: history cannot outlast the session limit.
    }
  }

  @Test fun manifest_rejectsUppercaseHash() {
    val tampered = validManifest().copy(
      files = validManifest().files.map {
        if (it.name == "tokens.json") it.copy(sha256 = sha.uppercase()) else it
      },
    )
    try {
      validateManifest(tampered)
      fail("expected uppercase SHA-256 to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: only lowercase hashes are accepted.
    }
  }

  @Test fun manifest_rejectsUnexpectedSize() {
    val tampered = validManifest().copy(
      files = validManifest().files.map {
        if (it.name == "tokens.json") it.copy(sizeBytes = 0) else it
      },
    )
    try {
      validateManifest(tampered)
      fail("expected out-of-bounds size to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: truncated or runaway files are rejected before copying.
    }
  }

  @Test fun manifest_rejectsUnknownFileAndWrongModel() {
    val renamed = validManifest().copy(
      files = validManifest().files.map {
        if (it.name == "tokens.json") it.copy(name = "extra.onnx") else it
      },
    )
    try {
      validateManifest(renamed)
      fail("expected unknown file name to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: only the four fixed file names are accepted.
    }
    try {
      validateManifest(validManifest().copy(modelId = "other-model"))
      fail("expected wrong modelId to be rejected")
    } catch (_: IllegalArgumentException) {
      // Expected: the manifest must describe the pinned mobile export.
    }
    assertEquals(4, validateManifest(validManifest()).size)
  }
}
