package com.stilltype.android.core

/**
 * Phase-preserving frame selection on a monotonic clock. Deadlines advance by a
 * fixed [intervalMs] from the previous deadline, so a 30 fps source keeps five of
 * every six frames (25 fps) instead of the fifteen a minimum gap from the last
 * accepted arrival would allow. A stall of a full interval or more re-anchors to
 * the current arrival; overdue slots are never replayed. Not thread-safe: use one
 * instance per analyzer binding on its single analysis thread.
 */
class FrameCadence(private val intervalMs: Long = 40L) {
    private var nextDeadlineMs: Long? = null

    fun shouldAccept(nowMs: Long): Boolean {
        val deadline = nextDeadlineMs
        if (deadline == null) {
            nextDeadlineMs = nowMs + intervalMs
            return true
        }
        if (nowMs < deadline) return false
        nextDeadlineMs = if (nowMs - deadline >= intervalMs) nowMs + intervalMs else deadline + intervalMs
        return true
    }
}
