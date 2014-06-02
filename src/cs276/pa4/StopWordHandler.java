package cs276.pa4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class StopWordHandler {

	// static String[] stopWords = { "a", "an", "and", "are", "as", "at", "be",
	// "but",
	// "by", "for", "if", "in", "into", "is", "it", "no", "not", "of",
	// "on", "or", "such", "that", "the", "their", "then", "there",
	// "these", "they", "this", "to", "was", "will", "with" };

	static String[] stopWords = { "a", "able", "about", "across", "after",
			"all", "almost", "also", "am", "among", "an", "and", "any", "are",
			"as", "at", "be", "because", "been", "but", "by", "can", "cannot",
			"could", "dear", "did", "do", "does", "either", "else", "ever",
			"every", "for", "from", "get", "got", "had", "has", "have", "he",
			"her", "hers", "him", "his", "how", "however", "i", "if", "in",
			"into", "is", "it", "its", "just", "least", "let", "like",
			"likely", "may", "me", "might", "most", "must", "my", "neither",
			"no", "nor", "not", "of", "off", "often", "on", "only", "or",
			"other", "our", "own", "rather", "said", "say", "says", "she",
			"should", "since", "so", "some", "than", "that", "the", "their",
			"them", "then", "there", "these", "they", "this", "tis", "to",
			"too", "twas", "us", "wants", "was", "we", "were", "what", "when",
			"where", "which", "while", "who", "whom", "why", "will", "with",
			"would", "yet", "you", "your" };

	public static String scrubStopWords(String str) {
		String resultStr = str;
		for (String stopWord : stopWords) {
			resultStr = (" " + str + " ").replaceAll(" " + stopWord + " ", " ")
					.replace("\\s+", " ").trim();
			str = resultStr;
		}
		return str.trim();
	}

	public static boolean isStopWord(String str) {
		boolean isStopWord = false;
		for (String stop_word : stopWords) {
			if (str.equals(stop_word)) {
				isStopWord = true;
				break;
			}
		}
		return isStopWord;
	}

	public static List<String> removeStopWords(List<String> queryWords) {
		List<String> qWords = new ArrayList<String>();
		
		for (String string : queryWords) {
			qWords.add(string);
		}
		for (String stop_word : stopWords) {
			if (qWords.contains(stop_word)) {
				qWords.remove(stop_word);
			}
		}
		return qWords;
	}

}
