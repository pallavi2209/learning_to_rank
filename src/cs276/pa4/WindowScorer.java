package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class WindowScorer{


//	public WindowScorer(Map<Query,List<Document>> queryDict) 
//	{
//		//super(queryDict);
////		handleSmallestWindow();
//	}
	
//	// /////////////weights///////////////////////////
//		static double urlweight = 9.0;
//		static double titleweight = 7.0;
//		static double bodyweight = 1.0;
//		static double headerweight = 7.5;
//		static double anchorweight = 1.2;
//
//		// /////bm25 specific weights///////////////
//		static double burl = 0.8;
//		static double btitle = 0.8;
//		static double bheader = 0.8;
//		static double bbody = 1.0;
//		static double banchor = 0.8;
//
//		static double k1 = 55.0;
//		static double pageRankLambda = 90.0d;
//		static double pageRankLambdaPrime = 1.0d;

		//////////////////////////////

	/////smallest window specifichyperparameters////////
	static double boostFactor1 = 0.1d;
	static double boostFactor2 = 0.5d;
	
//	
//	private void handleSmallestWindow() {
//		setParameters(urlweight, titleweight, bodyweight, headerweight,
//				anchorweight, burl, btitle, bheader, bbody, banchor, k1,
//				pageRankLambda, pageRankLambdaPrime);
//		calcAverageLengths();
//		
//	}

//	@Override
//	public double getSimScore(Document d, Query q) {
//		Map<String,Map<String, Double>> tfs = Util.getDocTermFreqs(d,q);
//		this.normalizeTFsBM25(tfs, d, q);
//		Map<String,Double> tfQuery = getQueryFreqs(q);
//		double netScore = getWindowScore(d, q, tfs, tfQuery);
//		return netScore;
//	}

	
	
//	public double getWindowScore(Document d, Query q,
//			Map<String, Map<String, Double>> tfs, Map<String, Double> tfQuery) {
//
//		double score = this.getNetScore(tfs, q, tfQuery, d);
//		double boost = getBoost(q, d);
//		double netScore = score * boost;
//		return netScore;
//	}

	public double getBoost(Query q, Document d){

		int smallestWindow = Integer.MAX_VALUE;
		List<String> queryTerms = q.words;
		ArrayList<String> fields = new ArrayList<String>();
		fields.add(Util.scrub(d.url));

		if (d.title != null) fields.add(d.title.toLowerCase());

		if (d.headers != null) {
			for (String header : d.headers) {
				fields.add(header);
			}
		}

		if (d.anchors != null) {
			for (String anchorText : d.anchors.keySet()) {
				fields.add(anchorText);
			}
		}

		for (String field : fields) {
			Map<String, List<Integer>> hits = getHits(field, q);
			int size = getWindowSize(hits, queryTerms);
			if (size < smallestWindow) smallestWindow = size;
		}

		if (d.body_hits != null) {
			int size = getWindowSize(d.body_hits, queryTerms);
			if (size < smallestWindow) smallestWindow = size;
		}
		
		List<String> queryTermsNoStop = StopWordHandler.removeStopWords(q.words);
		double boost = calculateBoost(smallestWindow, queryTermsNoStop.size());

		return boost;
	}


	private Map<String, List<Integer>> getHits(String field, Query q) {
		
		List<String> stemmedQterms = new ArrayList<String>();
		for (String qWord : q.words) {
			String stemmedWord = Util.stem(qWord);
			stemmedQterms.add(stemmedWord);
		}

		HashMap<String, List<Integer>> hits = new HashMap<String, List<Integer>>();

		String[] terms = field.split("\\s+");
		for (int i = 0; i < terms.length; i++) {
			String stemmedTerm = Util.stem(terms[i]);
			if (hits.containsKey(stemmedTerm)) {
				hits.get(stemmedTerm).add(i);
			}else{
				if(stemmedQterms.contains(stemmedTerm)){
					hits.put(stemmedTerm, new ArrayList<Integer>());
					hits.get(stemmedTerm).add(i);
				}
			}
		}

		return hits;
	}

	private int getWindowSize(Map<String, List<Integer>> hits, List<String> queryWords){
		ArrayList<Integer> positions = new ArrayList<Integer>();
		ArrayList<Iterator<Integer>> iterators = new ArrayList<Iterator<Integer>>();

		for (String term : queryWords) {
			if(StopWordHandler.isStopWord(term)){
				continue;
			}
			String stemmedTerm = Util.stem(term);
			if (!hits.containsKey(stemmedTerm)) return Integer.MAX_VALUE;
			iterators.add(hits.get(stemmedTerm).iterator());
		}

		for (Iterator<Integer> i : iterators) {
			positions.add(i.next());
		}

		int smallestWindow = Integer.MAX_VALUE;

		while (true) {
			int window = Collections.max(positions) - Collections.min(positions) +1;
			if (window < smallestWindow) smallestWindow = window;
			if (!advanceWindow(positions, iterators)) break;
		}

		return smallestWindow;
		
	}

	private boolean advanceWindow(List<Integer> positions, List<Iterator<Integer>> iterators) {
		int min = positions.get(0);
		int minIndex = 0;
		for (int i = 1; i < positions.size(); i++) {
			int x = positions.get(i);
			if (x < min) {
				x = min;
				minIndex = i;
			}
		}
		Iterator<Integer> iterator = iterators.get(minIndex);
		if (iterator.hasNext()) {
			positions.set(minIndex, iterator.next());
			return true;
		} else {
			return false;
		}

	}


	private double calculateBoost(int window, int length) {

		if(window == Integer.MAX_VALUE){
			return 1.0;
		}
		double boost = 1.0 +  boostFactor1/((double)(window-length) + boostFactor2);
		return boost;

	}
	
	public double[] getWindowScoreVector(Query q, Document d, Map<String, Double> dfs, Map<Query,List<Document>> queryDict){
		double[] result = {0.0, 0.0, 0.0, 0.0, 0.0};
		double boost = getBoost(q, d);
		
		BM25Scorer bmScorer = new BM25Scorer(queryDict);
		double[] bm25Vector = bmScorer.getBM25ScoreVector(q, d, dfs);
		
		for(int i = 0; i<bm25Vector.length; i++){
			result[i]= bm25Vector[i]*boost;
		}
//		CosineScorer cosScore = new CosineScorer();
//		double[] cosineVector = cosScore.getCosineScoreVector(q, d, dfs);
//		
//		for(int i = 0; i<cosineVector.length; i++){
//			result[i]= cosineVector[i]*boost;
//		}
		
		return result;
	}
	
}
