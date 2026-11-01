package com.stilltype.android.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

internal data class ComputeDiagnostics(val cpuNodes: Int, val nnapiNodes: Int, val complete: Boolean) {
    val label: String
        get() = when {
            !complete || cpuNodes + nnapiNodes == 0 -> "NNAPI requested; execution unverified"
            nnapiNodes > 0 && cpuNodes > 0 -> "NNAPI execution + CPU"
            nnapiNodes > 0 -> "NNAPI execution"
            else -> "CPU"
        }
}

/** Counts executed ORT node events, not provider availability or session initialization. */
internal fun parseComputeProfile(profile: String): ComputeDiagnostics {
    val unverified = ComputeDiagnostics(0, 0, false)
    return try {
        val events = Json.parseToJsonElement(profile) as? JsonArray ?: return unverified
        var cpu = 0
        var nnapi = 0
        var complete = true
        for (element in events) {
            val event = element as? JsonObject ?: return unverified
            val category = (event["cat"] as? JsonPrimitive)?.contentOrNull ?: return unverified
            if (category != "Node") continue
            if ((event["ph"] as? JsonPrimitive)?.contentOrNull != "X") {
                complete = false
                continue
            }
            val args = event["args"] as? JsonObject
            when ((args?.get("provider") as? JsonPrimitive)?.contentOrNull) {
                "CPUExecutionProvider" -> cpu++
                "NnapiExecutionProvider" -> nnapi++
                else -> complete = false
            }
        }
        ComputeDiagnostics(cpu, nnapi, complete)
    } catch (_: Exception) {
        unverified
    }
}
