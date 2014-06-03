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

public class PairwiseAddedFeatures extends Learner{
	  private LibSVM model;
	  private Standardize filter = new Standardize();
	  public PairwiseAddedFeatures(boolean isLinearKernel){
	    try{
	      model = new LibSVM();
	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	    }
	  }
	  
	  public PairwiseAddedFeatures(double C, double gamma, boolean isLinearKernel){
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
	  
	  
	  private Instances standardize(Instances instances) throws Exception{
		  //Standardize filter = new Standardize();
		  filter.setInputFormat(instances);
		  Instances normalized = Filter.useFilter(instances, filter);
		  return normalized;
	  }

	  private double[] getScoreVector(Query q, Document d, Map<String, Double> dfs, Map<Query, List<Document>> queryDict) {
		  
		  int numTFTypes = Util.TFTYPES.length;
		  double[] result = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
//		  int matchPresent = getScoreWithoutBody(q,d, dfs, tfs) ? 1 :0;
		  BM25Scorer bmScorer = new BM25Scorer(queryDict);
		  WindowScorer wScorer = new WindowScorer();
//		  ExtraScorer extraScorer = new ExtraScorer();
		  result[numTFTypes] = bmScorer.getNetScore(q, d, dfs);
		  result[numTFTypes+1] = wScorer.getBoost(q, d);
		  result[numTFTypes+2] = (double)d.page_rank;
//		  result[numTFTypes+3] = extraScorer.calcURLrelevance(q.words, d.url);
		  int isPdf = 0;
		  if(d.url.endsWith("pdf")){
			  isPdf = 1;
		  }
		  result[numTFTypes+3] = (double)isPdf;
		  int isTilde = 0;
		  if(d.url.contains("~")){
			  isTilde = 1;
		  }
		  result[numTFTypes+4] = (double)isTilde;
		  result[numTFTypes+5] = (double)d.title.length();
		  int isID = 0;
		  if(d.url.contains("id=")||d.url.contains("ID=")){
			  isID = 1;
		  }
		  result[numTFTypes+6] = (double)isID;
//		  result[numTFTypes+3] = (double)matchPresent;
		  return result;
	  }

//	  private boolean getScoreWithoutBody(Query q, Document d,
//			Map<String, Double> dfs, Map<String, Map<String, Double>> tfs) {
//		  
//		  boolean matchPresent = false;
//		  if(d.title!=null && d.title.trim().equals(q.query)){
//			  matchPresent = true;
//		  }
//		  if(!matchPresent && d.headers!=null){
//			  for (String header : d.headers) {
//				if(header.trim().equals(q.query)){
//					matchPresent = true;
//				}
//			}
//		  }
////		  if(!matchPresent && d.anchors!=null){
////			  for (String anchor : d.anchors.keySet()) {
////				if(anchor.trim().equals(q.query)){
////					matchPresent = true;
////				}
////			}
////		  }
//		  
//		  return matchPresent;
//		  
//	}

	private double[] difference(double[] vec1, double[]vec2) {
	  	double[] result = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
			attributes.add(new Attribute("bm25_w"));
			attributes.add(new Attribute("window_w"));
			attributes.add(new Attribute("pagerank_w"));
//			attributes.add(new Attribute("url_rel_w"));
			attributes.add(new Attribute("isPdf_w"));
			attributes.add(new Attribute("isTilde_w"));
			attributes.add(new Attribute("title_length"));
			attributes.add(new Attribute("isID_w"));
//			attributes.add(new Attribute("isHeaderPresent"));
			attributes.add(new Attribute("classification", classes));
			dataset = new Instances("train_dataset", attributes, 0);
			normalized = new Instances("normalized", attributes, 0);
			
			/* Set last attribute as target */
			dataset.setClassIndex(dataset.numAttributes() - 1);
			
			/* query -> docs */
			Map<Query,List<Document>> queryDict = Util.loadTrainData(train_data_file);
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
					double[] vector = getScoreVector(q, d, idfs, queryDict);
					Instance inst = new DenseInstance(1.0, vector);
					normalized.add(inst);
					indexes.put(d.url.toString(), index++);
				}
				instanceIndexes.put(q.toString(), indexes);
			}
			normalized = standardize(normalized);

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
			attributes.add(new Attribute("bm25_w"));
			attributes.add(new Attribute("window_w"));
			attributes.add(new Attribute("pagerank_w"));
//			attributes.add(new Attribute("url_rel_w"));
			attributes.add(new Attribute("isPdf_w"));
			attributes.add(new Attribute("isTilde_w"));
			attributes.add(new Attribute("title_length"));
			attributes.add(new Attribute("isID_w"));
//			attributes.add(new Attribute("isHeaderPresent"));
			attributes.add(new Attribute("classification", classes));
			features = new Instances("test_dataset", attributes, 0);
			
			/* Set last attribute as target */
			features.setClassIndex(features.numAttributes() - 1);
			
			Map<Query,List<Document>> queryDict = Util.loadTrainData(test_data_file);
			
			int index = 0;
			for (Entry<Query, List<Document>> entry : queryDict.entrySet()) {
				Query q = entry.getKey();
				//testFeatures.index_map.put(q.query, new HashMap<String,Integer>());
				List<Document> docs = entry.getValue();
				Map<String, Integer> queryFeatures = new HashMap<String, Integer>();
				for (Document d : docs) {
					double[] instance = getScoreVector(q, d, idfs, queryDict);
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
