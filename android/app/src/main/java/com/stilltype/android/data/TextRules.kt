package com.stilltype.android.data

import com.stilltype.android.core.VocabularyWord

/**
 * Settled vocabulary semantics, matching the desktop engine exactly: exact
 * case-insensitive word/phrase mapping with the longest source phrase winning
 * at each position. The output is the entry's replacement when non-empty,
 * otherwise the source spelling as typed. Leading/trailing punctuation is
 * preserved and replacements are never re-scanned, so mappings cannot chain.
 * No fuzzy or phonetic correction is applied.
 */
object TextRules {
  fun applyVocabulary(text: String, words: List<VocabularyWord>): String {
    val pairs = vocabularyPairs(words)
    if (pairs.isEmpty() || text.isEmpty()) return text
    val ordered = pairs.sortedByDescending { it.source.size }
    // Kotlin split on runs plus the empty filter reproduces Python's str.split().
    val tokens = text.split(Regex("\\s+")).filter { it.isNotEmpty() }
    val cores = tokens.map { splitEdge(it).second.lowercase() }
    val out = ArrayList<String>(tokens.size)
    var i = 0
    while (i < tokens.size) {
      var matched = false
      for ((source, output) in ordered) {
        if (source.isEmpty() || i + source.size > tokens.size) continue
        val window = cores.subList(i, i + source.size)
        if (window.any { it.isEmpty() }) continue
        if (window == source) {
          val (lead, _, _) = splitEdge(tokens[i])
          val (_, _, trail) = splitEdge(tokens[i + source.size - 1])
          out += lead + output + trail
          i += source.size
          matched = true
          break
        }
      }
      if (!matched) {
        out += tokens[i]
        i++
      }
    }
    return out.joinToString(" ")
  }

  private data class VocabularyPair(val source: List<String>, val output: String)

  private fun vocabularyPairs(words: List<VocabularyWord>): List<VocabularyPair> {
    val pairs = ArrayList<VocabularyPair>()
    val seen = HashSet<List<String>>()
    for (entry in words) {
      val source = entry.word.trim()
      if (source.isEmpty()) continue
      val key = source.lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
      // A sooner duplicate source keeps the first output.
      if (key.isEmpty() || !seen.add(key)) continue
      pairs += VocabularyPair(key, entry.replacement.trim().ifEmpty { source })
    }
    return pairs
  }

  private fun splitEdge(token: String): Triple<String, String, String> {
    var start = 0
    while (start < token.length && !isWordChar(token[start])) start++
    var end = token.length
    while (end > start && !isWordChar(token[end - 1])) end--
    return Triple(token.substring(0, start), token.substring(start, end), token.substring(end))
  }

  /** Python's \W (unicode) treats letters, digits and underscore as word chars. */
  private fun isWordChar(c: Char): Boolean = c.isLetterOrDigit() || c == '_'
}
