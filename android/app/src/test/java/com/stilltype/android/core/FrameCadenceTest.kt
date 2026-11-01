package com.stilltype.android.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameCadenceTest {
    private fun acceptedPerSecond(fps: Int, seconds: Int): Int {
        val cadence = FrameCadence()
        return (0 until fps * seconds).count { cadence.shouldAccept(it * 1000L / fps) }
    }

    @Test fun steadyHighRateInput_isCappedNearTwentyFivePerSecondNotLower() {
        // A 40 ms gap from the last accepted arrival would give 15 and 20 here.
        assertTrue(acceptedPerSecond(fps = 30, seconds = 6) in 149..151)
        assertTrue(acceptedPerSecond(fps = 60, seconds = 6) in 149..151)
    }

    @Test fun inputAtOrBelowTargetRate_keepsEveryFrame() {
        assertEquals(25 * 4, acceptedPerSecond(fps = 25, seconds = 4))
        assertEquals(24 * 4, acceptedPerSecond(fps = 24, seconds = 4))
    }

    @Test fun duplicateAndEarlyArrivals_areRejected() {
        val cadence = FrameCadence()
        assertTrue(cadence.shouldAccept(1_000))
        assertFalse(cadence.shouldAccept(1_000))
        assertFalse(cadence.shouldAccept(1_039))
        assertTrue(cadence.shouldAccept(1_040))
    }

    @Test fun longStall_acceptsOneFrameAndDoesNotReplayOverdueSlots() {
        val cadence = FrameCadence()
        assertTrue(cadence.shouldAccept(0))
        assertTrue(cadence.shouldAccept(10_000))
        assertFalse(cadence.shouldAccept(10_001))
        assertFalse(cadence.shouldAccept(10_039))
        assertTrue(cadence.shouldAccept(10_040))
    }
}
