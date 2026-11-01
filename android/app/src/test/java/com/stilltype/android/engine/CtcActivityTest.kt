package com.stilltype.android.engine

import java.nio.FloatBuffer
import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CtcActivityTest {
    @Test fun weakActivity_requiresDistinctCollapsedEmissionsAndFiniteOutput() {
        val tokens = listOf("<blank>", "a", "b", "<eos>")
        fun frame(id: Int, probability: Float = 0.8f): List<Float> =
            List(tokens.size) { if (it == id) ln(probability) else -10f }
        fun decode(vararg frames: List<Float>) = decodeCtc(
            FloatBuffer.wrap(frames.flatMap { it }.toFloatArray()), frames.size, tokens, 0, 3,
        )

        val repeated = decode(frame(1), frame(1))
        assertEquals("a", repeated.text)
        assertFalse(repeated.hasActivityEvidence())
        val separated = decode(frame(1), frame(0), frame(1))
        assertEquals("aa", separated.text)
        assertTrue(separated.hasActivityEvidence())
        assertFalse(decode(frame(1, 0.2f), frame(2, 0.2f)).hasActivityEvidence())
        val bad = frame(1).toMutableList().apply { this[2] = Float.NaN }
        try {
            decode(bad)
            throw AssertionError("Non-finite output must fail recognition")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("Non-finite"))
        }
    }
}
