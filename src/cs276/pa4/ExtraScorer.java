package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;


public class ExtraScorer {
	
	public final Character[] alphabet = {'s',' ','-'};
	public void computeExhaustiveOneEdits(String query,
			Set<String> allCandidates) {
		// Deletion
		for (int idx = 0; idx < query.length(); ++idx)
			allCandidates.add(query.substring(0, idx)
					+ query.substring(idx + 1));

		// adjacent
		for (int idx = 0; idx < query.length() - 1; ++idx)
			allCandidates.add(query.substring(0, idx)
					+ query.substring(idx + 1, idx + 2)
					+ query.substring(idx, idx + 1) + query.substring(idx + 2));

		// replacements
		for (int idx = 0; idx < query.length(); ++idx) {
			for (int iAlpha = 0; iAlpha < alphabet.length; iAlpha++) {
				char character = alphabet[iAlpha].charValue();
				allCandidates.add(query.substring(0, idx)
						+ String.valueOf(character) + query.substring(idx + 1));
			}
		}

		// insertions
		for (int idx = 0; idx <= query.length(); ++idx) {
			for (int iAlpha = 0; iAlpha < alphabet.length; iAlpha++) {
				char character = alphabet[iAlpha].charValue();
				allCandidates.add(query.substring(0, idx)
						+ String.valueOf(character) + query.substring(idx));
			}
		}
	}
	
	private double calcUniqueness(Query q, Map<String, Map<String, Double>> tfs) {
		double uniquenessQueryTermsInDoc;
		int numQueryWords = q.words.size();
		List<String> qTermsInDoc = new ArrayList<String>();
		for (Entry<String,Map<String, Double>> entry : tfs.entrySet()) {
			for (Entry<String, Double> mapEntry : entry.getValue().entrySet()) {
				if(!qTermsInDoc.contains(mapEntry.getKey())){
					qTermsInDoc.add(mapEntry.getKey());
				}
			}
		}
		int numQTermsInDoc = qTermsInDoc.size();
		double numUniqueTerms = (double)numQTermsInDoc/(double)numQueryWords;
		uniquenessQueryTermsInDoc = numUniqueTerms;
		return uniquenessQueryTermsInDoc;
	}
	
	public List<String> checkUnigramsInQuery(String candidate, List<String> query_words, List<String> qTermsInUrl) {
		String[] termsCand= candidate.split(" ");
		//check if all unigrams of this candidate are present in our query, then add to candidates
		for (int j = 0; j < termsCand.length; j++) {
			String currentTermOfQuery = termsCand[j];
			if(query_words.contains(Util.stem(currentTermOfQuery))){
				if(!qTermsInUrl.contains(currentTermOfQuery)){
					qTermsInUrl.add(currentTermOfQuery);
				}
			}
		}
		return qTermsInUrl;
	}
	
	public double calcURLrelevance(List<String> queryWords, String url) {
		
		Map<String, Integer> qTermsInUrl = new HashMap<String, Integer>();
		int numMatch = 0;
		String[] strPartial = url.split("/+");
		
		List<String> stemmedQwords = new ArrayList<String>();
		for (String string : queryWords) {
			stemmedQwords.add(Util.stem(string));
		}
		
		for (String string : strPartial) {
			List<String> qTermsInPartialUrl= new ArrayList<String>();
			String strUrlpart = Util.scrub(string);
			Set<String> allCandidates = new HashSet<String>();
			allCandidates.add(strUrlpart);
			computeExhaustiveOneEdits(strUrlpart, allCandidates);
			for (String cand : allCandidates) {
				checkUnigramsInQuery(cand, queryWords, qTermsInPartialUrl);
			}
			for (String term : qTermsInPartialUrl) {
				int count = 1;
				if(qTermsInUrl.containsKey(term)){
					count = qTermsInUrl.get(term)+1;
				}
				qTermsInUrl.put(term, count);
			}
		}
		for (int num : qTermsInUrl.values()) {
			numMatch+=num;
		}
		
		double relScore = 1.0d + (double)numMatch*0.01d;
		return relScore;
	}

}
