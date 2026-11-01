package com.stilltype.android.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class ComputeDiagnosticsTest {
    @Test fun cpuNodeEvents_reportCpuRatherThanRequestedProvider() {
        val profile = """[{"cat":"Session","ph":"X"},{"cat":"Node","ph":"X","args":{"provider":"CPUExecutionProvider"}}]"""
        assertEquals(ComputeDiagnostics(1, 0, true), parseComputeProfile(profile))
        assertEquals("CPU", parseComputeProfile(profile).label)
    }

    @Test fun mixedNodeEvents_reportNnapiAndCpu() {
        val profile = """[{"cat":"Node","ph":"X","args":{"provider":"NnapiExecutionProvider"}},{"cat":"Node","ph":"X","args":{"provider":"CPUExecutionProvider"}}]"""
        assertEquals(ComputeDiagnostics(1, 1, true), parseComputeProfile(profile))
        assertEquals("NNAPI execution + CPU", parseComputeProfile(profile).label)
    }

    @Test fun incompleteOrMalformedProfiles_neverClaimExecution() {
        for (profile in listOf(
            "", "[]", "[{", "[{\"cat\":\"Node\",\"ph\":\"X\",\"args\":{}}]",
            """[{"cat":"Node","ph":"X","args":{"provider":"NnapiExecutionProvider"}},{"cat":"Node","args":{}}]""",
        )) {
            assertEquals("NNAPI requested; execution unverified", parseComputeProfile(profile).label)
        }
    }
}
