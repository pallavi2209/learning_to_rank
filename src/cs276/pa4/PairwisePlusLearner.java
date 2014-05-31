package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwisePlusLearner extends Learner {
	  private LibSVM model;
	  private Standardize filter = new Standardize();
	  Map<Query,List<Document>> queryDict;
	  public PairwisePlusLearner(boolean isLinearKernel, String train_data_file){
	    try{
	      model = new LibSVM();
	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	    }
	    try {
			this.queryDict = Util.loadTrainData(train_data_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    this.calcAverageLengths();
	  }
	  
	  public PairwisePlusLearner(double C, double gamma, boolean isLinearKernel){
	    try{
	      model = new LibSVM();
	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    	
	    model.setCost(C);
	    model.setGamma(gamma); // only matter for RBF kernel
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	    }
	  }
		// /////////////weights///////////////////////////
//		double urlweight = 8.0;
//		double titleweight = 6.0;
//		double bodyweight = 1.0;
//		double headerweight = 7.5;
//		double anchorweight = 1.2;
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
		//double pageRankLambda = 65.0d;
		//double pageRankLambdaPrime = 1.0d;

	    ////////////bm25 data structures--feel free to modify ////////
	    
	    Map<Document,Map<String,Double>> lengths;
	    Map<String,Double> avgLengths;
	    //Map<Document,Double> pagerankScores;
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
	    
//		private double calcPageRankFactor(Document doc) {
//			double factor = Math.log10(pageRankLambdaPrime + (double) doc.page_rank);
//			return factor;
//		}
		
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
	    public void calcAverageLengths()
	    {
	    	weights = new HashMap<String, Double>();
	    	bvals = new HashMap<String, Double>();
	    	lengths = new HashMap<Document,Map<String,Double>>();
	    	avgLengths = new HashMap<String,Double>();
	    	//pagerankScores = new HashMap<Document,Double>();

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

	    	
			for (Entry<Query, List<Document>> queryDictEntry : this.queryDict.entrySet()) {
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
//					if (doc.page_rank > 0) {
//						pagerankScores.put(doc, calcPageRankFactor(doc));
//					} else {
//						pagerankScores.put(doc, 0.0);
//					}
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
	  
	  private Instances standardize(Instances instances) throws Exception{
		  //Standardize filter = new Standardize();
		  filter.setInputFormat(instances);
		  Instances normalized = Filter.useFilter(instances, filter);
		  return normalized;
	  }

	  private double[] lengthNormalize(double[] vec) {
	  	double total = 0.0;
	  	for (double d : vec) total += d * d;
	  	//if (total == 0) return vec;
	  	double x = Math.sqrt(total);
	  	for (int i = 0; i < vec.length; i++) {
	  		vec[i] = vec[i] / x;
	  	}
	  	return vec;
	  }

	  
	  private double[] getScoreVector(Query q, Document d, Map<String, Double> dfs) {
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
	  
	  
		// do bm25 normalization
		public void normalizeTFsBM25(Map<String, Map<String, Double>> tfs, Document d,
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
	  
	  private double[] getBM25ScoreVector(Query q, Document d, Map<String, Double> dfs) {
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
			  	result[i] = score;
		  }
		  return result;
	  }

	  private double[] difference(double[] vec1, double[]vec2) {
	  	double[] result = {0.0, 0.0, 0.0, 0.0, 0.0};
	  	for (int i = 0; i < result.length; i++) {
	  		result[i] += vec1[i];
	  		result[i] -= vec2[i];
	  	}
	  	return result;
	  }
	  
		@Override
		public Instances extract_train_features(String train_data_file,
				String train_rel_file, Map<String, Double> idfs) throws Exception{
			
			Instances dataset = null;
			Instances normalized = null;
			
			List<String> classes = new ArrayList<String>();
			classes.add("1");
			classes.add("-1");
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("body_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("classification", classes));
			dataset = new Instances("train_dataset", attributes, 0);
			normalized = new Instances("normalized", attributes, 0);
			
			/* Set last attribute as target */
			dataset.setClassIndex(dataset.numAttributes() - 1);
			
			/* query -> docs */
			//queryDict = Util.loadTrainData(train_data_file);
			/* query -> (url -> score) */
			Map<String, Map<String, Double>> relevanceScores = Util.loadRelData(train_rel_file);
			/* query -> (url -> index) */
			Map<String, Map<String, Integer>> instanceIndexes = new HashMap<String, Map<String, Integer>>();

			int index = 0;
			for (Entry<Query,List<Document>> entry : queryDict.entrySet()) {
				Query q = entry.getKey();
				List<Document> docs = entry.getValue();
				Map<String, Integer> indexes = new HashMap<String, Integer>();
				for (Document d : docs) {
					double[] vector = getBM25ScoreVector(q, d, idfs);
					Instance inst = new DenseInstance(1.0, vector);
					normalized.add(inst);
					indexes.put(d.url.toString(), index++);
				}
				instanceIndexes.put(q.toString(), indexes);
			}

			System.out.println(normalized.numInstances());

			normalized = standardize(normalized);

			System.out.println(normalized.numInstances());
			

			for (Entry<Query, List<Document>> entry : queryDict.entrySet()) {
				Query q = entry.getKey();
				List<Document> docs = entry.getValue();
				for (int i = 0; i < docs.size(); i++) {
					Document d1 = docs.get(i);
					double rel1 = relevanceScores.get(q.toString()).get(d1.url.toString()); 
					for (int j = 0; j < docs.size(); j++) {
						Document d2 = docs.get(j);
						double rel2 = relevanceScores.get(q.toString()).get(d2.url.toString()); 
						if (i == j | rel1 == rel2) continue;
						double[] vector1 = normalized.get(instanceIndexes.get(q.toString()).get(d1.url.toString())).toDoubleArray();
						double[] vector2 = normalized.get(instanceIndexes.get(q.toString()).get(d2.url.toString())).toDoubleArray();


						double[] diff = difference(vector1, vector2);
						Instance inst = new DenseInstance(1.0, diff);
						String dataclass = (rel1 > rel2) ? "1" : "-1";

						inst.insertAttributeAt(inst.numAttributes());
						inst.setValue(dataset.attribute(inst.numAttributes() - 1), dataclass);
						dataset.add(inst);
					}
				}
			}
			
			return dataset;
		}

		@Override
		public Classifier training(Instances dataset) throws Exception{
			model.buildClassifier(dataset);
			return model;
		}

		@Override
		public TestFeatures extract_test_features(String test_data_file,
				Map<String, Double> idfs) throws Exception{
			
			Instances features = null;
			TestFeatures testFeatures = new TestFeatures();
			testFeatures.index_map = new HashMap<String, Map<String, Integer>>();
			
			List<String> classes = new ArrayList<String>();
			classes.add("1");
			classes.add("-1");
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("body_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("classification", classes));
			features = new Instances("test_dataset", attributes, 0);
			
			/* Set last attribute as target */
			features.setClassIndex(features.numAttributes() - 1);
			
			//Map<Query,List<Document>> queryDict = Util.loadTrainData(test_data_file);
			
			int index = 0;
			for (Entry<Query, List<Document>> entry : queryDict.entrySet()) {
				Query q = entry.getKey();
				//testFeatures.index_map.put(q.query, new HashMap<String,Integer>());
				List<Document> docs = entry.getValue();
				Map<String, Integer> queryFeatures = new HashMap<String, Integer>();
				for (Document d : docs) {
					double[] instance = getBM25ScoreVector(q, d, idfs);
					Instance inst = new DenseInstance(1.0, instance);
					inst.insertAttributeAt(inst.numAttributes());
					inst.setDataset(features);
					inst.setClassMissing();
					features.add(inst);
					//testFeatures.index_map.get(q.query).put(d.url.toString(), index++);
					queryFeatures.put(d.url.toString(), index++);
				}
				testFeatures.index_map.put(q.query, queryFeatures);
			}
			System.out.println("Num Intances: " + features.numInstances() + " Index: " + index);
			testFeatures.features = standardize(features);
			return testFeatures;
			
		}

		
		private int compareDocs(TestFeatures tf, Classifier model, String q, String url1, String url2)  {
			double[] vector1 = tf.features.get(tf.index_map.get(q).get(url1)).toDoubleArray();
			double[] vector2 = tf.features.get(tf.index_map.get(q).get(url2)).toDoubleArray();
			double[] diff = difference(vector1, vector2);
			Instance inst = new DenseInstance(1.0, diff);
			inst.setDataset(tf.features);
			double classification = 1.0;
			try {
				classification = model.classifyInstance(inst);
				//System.out.println(classification);
			} catch (Exception e) {
				System.err.println("An error occured while classifying.");
			}
			return (classification > 0.0) ? 1 : -1;
		}

		@Override
		public Map<String, List<String>> testing(final TestFeatures tf,
			final Classifier model) throws Exception{
			Map<String, List<String>> allRankings = new HashMap<String, List<String>>();
			for (Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
				final String q = entry.getKey();
				List<String> rankings = new ArrayList<String>();
				for (String url : entry.getValue().keySet()) {
					rankings.add(url);
				}
				Collections.sort(rankings, new Comparator<String>() {
					@Override
					public int compare(String url1, String url2)
					{
						return compareDocs(tf, model, q, url1, url2);
					}	
				});
				allRankings.put(q, rankings);
			}
			return allRankings;
		}

}
