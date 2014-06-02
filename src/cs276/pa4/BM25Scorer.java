package cs276.pa4;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;


public class BM25Scorer {
	
	Map<Query, List<Document>> queryDict;
	
	public BM25Scorer(Map<Query, List<Document>> queryDictionary){
		this.queryDict = queryDictionary;
		this.calcAverageLengths();
	}
	 
	
	// /////////////weights///////////////////////////
//	double urlweight = 8.0;
//	double titleweight = 6.0;
//	double bodyweight = 1.0;
//	double headerweight = 7.5;
//	double anchorweight = 1.2;
  	double urlweight = 1.0;
	double titleweight = 1.0;
	double bodyweight = 1.0;
	double headerweight = 1.0;
	double anchorweight = 1.0;

	// /////bm25 specific weights///////////////
	double burl = 0.6;
	double btitle = 0.6;
	double bheader = 0.6;
	double bbody = 1.0;
	double banchor = 0.6;

	double k1 = 55.0;
//	double pageRankLambda = 65.0d;
//	double pageRankLambdaPrime = 1.0d;

    ////////////bm25 data structures--feel free to modify ////////
    
    Map<Document,Map<String,Double>> lengths;
    Map<String,Double> avgLengths;
//    Map<Document,Double> pagerankScores;
    Map<String, Double> weights;
    Map<String, Double> bvals;
    
    //////////////////////////////////////////
    
    
    
    //simple helper function to computer total length of anchor text
    double calcAnchorsLength(Map<String, Integer> anchors) {
    	double length = 0.0;
    	for (Entry<String, Integer> anchor : anchors.entrySet()) {
    		length += anchor.getValue() * anchor.getKey().split("\\s+").length;
    	}
    	return length;
    }

    double calcHeadersLength(List<String> headers) {
    	double length = 0.0;
    	for (String header : headers) {
    		length += header.split("\\s+").length;
    	}
    	return length;
    }

    double calcTitleLength(String title) {
    	return (double )title.split("\\s+").length;
    }
//    
//	private double calcPageRankFactor(Document doc) {
//		double factor = Math.log10(pageRankLambdaPrime + (double) doc.page_rank);
//		return factor;
//	}
	
	private double zoneLength(Document d, String zone) {
		if (zone.equals("url"))
			return d.url.length();
		if (zone.equals("body"))
			return d.body_length;
		if (zone.equals("title")) {
			return (d.title != null) ? d.title.length() : 0;
		}
		if (zone.equals("header")) {
			return (d.headers != null) ? calcHeadersLength(d.headers) : 0;
		}
		if (zone.equals("anchor")) {
			return (d.anchors != null) ? calcAnchorsLength(d.anchors) : 0;
		}
		return 0;
	}
  
    //sets up average lengths for bm25, also handles pagerank
    private void calcAverageLengths()
    {
    	weights = new HashMap<String, Double>();
    	bvals = new HashMap<String, Double>();
    	lengths = new HashMap<Document,Map<String,Double>>();
    	avgLengths = new HashMap<String,Double>();
//    	pagerankScores = new HashMap<Document,Double>();

    	weights.put("url", urlweight);
    	weights.put("title", titleweight);
    	weights.put("body", bodyweight);
    	weights.put("header", headerweight);
    	weights.put("anchor", anchorweight);

    	bvals.put("url", burl);
    	bvals.put("title", btitle);
    	bvals.put("header", bheader);
    	bvals.put("body", bbody);
    	bvals.put("anchor", banchor);

    	
		for (Entry<Query, List<Document>> queryDictEntry : queryDict.entrySet()) {
			for (Document docEntry : queryDictEntry.getValue()) {
				String url = docEntry.url;
				Document doc = docEntry;
				Map<String, Double> docLengths = new HashMap<String, Double>();
				docLengths.put("url", (double)url.length());
				if (doc.title != null) docLengths.put("title", calcTitleLength(doc.title));
				docLengths.put("body", (double)doc.body_length);
				if (doc.headers != null) docLengths.put("header", calcHeadersLength(doc.headers));
				if (doc.anchors != null) docLengths.put("anchor", calcAnchorsLength(doc.anchors));
				lengths.put(doc, docLengths);
//				if (doc.page_rank > 0) {
//					pagerankScores.put(doc, calcPageRankFactor(doc));
//				} else {
//					pagerankScores.put(doc, 0.0);
//				}
			}
		}

		for (Map<String, Double> zoneLengths : lengths.values()) {
			for (Entry<String, Double> zoneLength : zoneLengths.entrySet()) {
				String zone = zoneLength.getKey();
				Double length = zoneLength.getValue();
				if (avgLengths.containsKey(zone)) {
					avgLengths.put(zone, length / lengths.size() + avgLengths.get(zone));
				} else {
					avgLengths.put(zone, length / lengths.size());
				}
			}
		}

		// normalize avgLengths
		for (String tfType : Util.TFTYPES) {
			avgLengths.put(tfType, avgLengths.get(tfType) * weights.get(tfType));
		}

	}

	// do bm25 normalization
	protected void normalizeTFsBM25(Map<String, Map<String, Double>> tfs, Document d,
			Query q) {
		for (String tfType : Util.TFTYPES) {
			if (!tfs.containsKey(tfType))
				continue;
			Double Bz = (1 - bvals.get(tfType)) + bvals.get(tfType)
					* zoneLength(d, tfType) / avgLengths.get(tfType);
			for (Entry<String, Double> tf : tfs.get(tfType).entrySet()) {
				tfs.get(tfType).put(tf.getKey(),
						tf.getValue() * weights.get(tfType) / Bz);
			}
		}
	}
  
	public double getNetScore(Query q, Document d, Map<String, Double> dfs) {
		double score = 0.0;

		 Map<String, Map<String, Double>> tfs = Util.getDocTermFreqs(d, q);
		  this.normalizeTFsBM25(tfs, d, q);

		  for (int i = 0; i < Util.TFTYPES.length; i++) {
			  	Map<String, Double> tf = tfs.get(Util.TFTYPES[i]);
				double zone_tfs[] = new double[q.words.size()];
			  	for (int j = 0; j < q.words.size(); j++) {
			  		String term = q.words.get(j);
			  		double df = dfs.containsKey(term) ? dfs.get(term) + 1.0 : 1.0;
			  		double idf = Math.log10((Util.totFiles + 1.0)/df);
			  		double subLinearTf = (tf != null && tf.containsKey(term)) ? Math.log10(tf.get(term)) + 1.0 : 0.0;
			  		double bm25Score = idf * (k1 + 1) * subLinearTf / (k1 + subLinearTf);
			  		zone_tfs[j] = bm25Score;
			  	}
			  	double score_zone = 0.0;
			  	for (int j = 0; j < zone_tfs.length; j++) {
			  		score_zone += zone_tfs[j];
			  	}
			  	score+=score_zone;
		  }
		  
//		  score += pageRankLambda * pagerankScores.get(d);
		  return score;
	}

	
	  public  double[] getBM25ScoreVector(Query q, Document d, Map<String, Double> dfs) {
		  double[] result = {0.0, 0.0, 0.0, 0.0, 0.0};
		  Map<String, Map<String, Double>> tfs = Util.getDocTermFreqs(d, q);
		  this.normalizeTFsBM25(tfs, d, q);

		  for (int i = 0; i < Util.TFTYPES.length; i++) {
			  	Map<String, Double> tf = tfs.get(Util.TFTYPES[i]);
				double zone_tfs[] = new double[q.words.size()];
			  	for (int j = 0; j < q.words.size(); j++) {
			  		String term = q.words.get(j);
			  		double df = dfs.containsKey(term) ? dfs.get(term) + 1.0 : 1.0;
			  		double idf = Math.log10((Util.totFiles + 1.0)/df);
			  		double subLinearTf = (tf != null && tf.containsKey(term)) ? Math.log10(tf.get(term)) + 1.0 : 0.0;
			  		double bm25Score = idf * (k1 + 1) * subLinearTf / (k1 + subLinearTf);
			  		zone_tfs[j] = bm25Score;
			  	}
			  	double score = 0.0;
			  	for (int j = 0; j < zone_tfs.length; j++) {
			  		score += zone_tfs[j];
			  	}
//			  	score += pageRankLambda * pagerankScores.get(d);
			  	result[i] = score;
		  }
		  return result;
	  }

	
}
