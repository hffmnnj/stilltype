package com.stilltype.android.data

import com.stilltype.android.core.VocabularyWord
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Test

/** Guards the settled desktop vocabulary semantics against regressions. */
class TextRulesTest {
  private fun word(word: String, replacement: String = ""): VocabularyWord =
    VocabularyWord(id = UUID.randomUUID().toString(), word = word, replacement = replacement)

  @Test fun phrase_replacement_isCaseInsensitive() {
    val out = TextRules.applyVocabulary(
      "I use still type daily",
      listOf(word("still type", "StillType")),
    )
    assertEquals("I use StillType daily", out)
  }

  @Test fun longestSourcePhrase_winsAndPunctuationIsPreserved() {
    val words = listOf(word("site", "PLACE"), word("bostic site", "Bostic Site"))
    assertEquals("the Bostic Site.", TextRules.applyVocabulary("the BOSTIC SITE.", words))
  }

  @Test fun emptyReplacement_fallsBackToSourceSpelling() {
    assertEquals(
      "I like colour",
      TextRules.applyVocabulary("I like COLOUR", listOf(word("colour"))),
    )
  }

  @Test fun replacements_areNeverChainedIntoOtherMappings() {
    val words = listOf(word("teh", "the"), word("the", "THE"))
    assertEquals("the cat", TextRules.applyVocabulary("teh cat", words))
  }

  @Test fun duplicateSource_keepsFirstOutput() {
    val words = listOf(word("still type", "StillType"), word("Still Type", "Other"))
    assertEquals("StillType rocks", TextRules.applyVocabulary("still type rocks", words))
  }
}
