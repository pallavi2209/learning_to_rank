package cs276.pa4;

import java.util.Map;

public class CosineScorer {

	  public double[] getCosineScoreVector(Query q, Document d, Map<String, Double> dfs) {
		  double[] result = {0.0, 0.0, 0.0, 0.0, 0.0};
		  Map<String, Map<String, Double>> tfs = Util.getDocTermFreqs(d, q);
		  for (int i = 0; i < Util.TFTYPES.length; i++) {
			  	Map<String, Double> tf = tfs.get(Util.TFTYPES[i]);
				double zone_tfs[] = new double[q.words.size()];
			  	for (int j = 0; j < q.words.size(); j++) {
			  		String term = q.words.get(j);
			  		double df = dfs.containsKey(term) ? dfs.get(term) + 1.0 : 1.0;
			  		double idf = Math.log10((Util.totFiles + 1.0)/df);
			  		double tfidf = (tf != null && tf.containsKey(term)) ? (Math.log10(tf.get(term)) + 1.0) * idf: 0.0;
			  		zone_tfs[j] = tfidf;
			  	}
			  	double score = 0.0;
			  	for (int j = 0; j < zone_tfs.length; j++) {
			  		score += zone_tfs[j];
			  	}
			  	result[i] = score;
		  }
		  return result;
	  }
	
}
