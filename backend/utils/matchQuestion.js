const QUESTIONS = require("../data/questions");

function tokenize(text) {
  return new Set(text.toLowerCase().match(/[a-z]+/g) || []);
}

function stringSimilarity(a, b) {
  if (a === b) return 1;
  if (a.length < 2 || b.length < 2) return 0;

  const bigrams = (s) => {
    const map = new Map();
    for (let i = 0; i < s.length - 1; i++) {
      const bg = s.substring(i, i + 2);
      map.set(bg, (map.get(bg) || 0) + 1);
    }
    return map;
  };

  const bigramsA = bigrams(a);
  const bigramsB = bigrams(b);

  let intersection = 0;
  for (const [bg, count] of bigramsA) {
    if (bigramsB.has(bg)) {
      intersection += Math.min(count, bigramsB.get(bg));
    }
  }

  const totalA = [...bigramsA.values()].reduce((sum, c) => sum + c, 0);
  const totalB = [...bigramsB.values()].reduce((sum, c) => sum + c, 0);

  return (2 * intersection) / (totalA + totalB);
}

const MATCH_THRESHOLD = 0.35;

function matchQuestion(spokenText) {
  if (!spokenText) return null;

  const spokenTokens = tokenize(spokenText);
  let bestMatch = null;
  let bestScore = 0;

  for (const question of QUESTIONS) {
    const keywordHits = question.keywords.filter((k) => spokenTokens.has(k));
    const keywordScore = Math.min(1.0, keywordHits.length / 2);
    const seqScore = stringSimilarity(spokenText, question.canonical);
    const combined = 0.6 * keywordScore + 0.4 * seqScore;

    if (keywordHits.length > 0 && combined > bestScore) {
      bestMatch = question;
      bestScore = combined;
    }
  }

  if (bestMatch && bestScore >= MATCH_THRESHOLD) {
    return { question: bestMatch, score: bestScore };
  }
  return null;
}

module.exports = matchQuestion;