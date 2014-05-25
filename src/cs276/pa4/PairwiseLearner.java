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
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  private boolean linear = false;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      linear = true;
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
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
  
  private double[] extractData(Query q, Document d1, Document d2, double rel1, double rel2, Map<String, Double> dfs) {
	  double relevance = (rel1 - rel1 > 0) ? 1.0 : -1.0;
	  double[] result = {0.0, 0.0, 0.0, 0.0, 0.0, relevance};
	  Map<String, Map<String, Double>> tfs1 = Util.getDocTermFreqs(d1, q);
	  Map<String, Map<String, Double>> tfs2 = Util.getDocTermFreqs(d2, q);
	  
	  for (int i = 0; i < Util.TFTYPES.length; i++) {
	  	Map<String, Double> tf1 = tfs1.get(Util.TFTYPES[i]);
	  	Map<String, Double> tf2 = tfs2.get(Util.TFTYPES[i]);
	  	
	  	for (String term : q.words) {
	  		double df = dfs.containsKey(term) ? dfs.get(term) + 1.0 : 1.0;
	  		double idf = Math.log10((Util.totFiles + 1.0)/df);
	  		if (tf1 != null && tf1.containsKey(term)) result[i] += tf1.get(term) * idf;
	  		if (tf2 != null && tf2.containsKey(term)) result[i] -= tf2.get(term) * idf;
	  	}
	  }
	  return result;
  }
  
  private Instances standardize(Instances instances) throws Exception{
	  Standardize filter = new Standardize();
	  filter.setInputFormat(instances);
	  Instances normalized = Filter.useFilter(instances, filter);
	  NumericToNominal ntn = new NumericToNominal();
	  ntn.setInputFormat(normalized);
	  return Filter.useFilter(normalized, ntn);
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception{
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		Map<Query,List<Document>> queryDict = Util.loadTrainData(train_data_file);
		/* query -> (url -> score) */
		Map<String, Map<String, Double>> relevanceScores = Util.loadRelData(train_rel_file);
		
		for (Entry<Query, List<Document>> entry : queryDict.entrySet()) {
			Query q = entry.getKey();
			List<Document> docs = entry.getValue();
			for (int i = 0; i < docs.size(); i++) {
				Document d1 = docs.get(i);
				double rel1 = relevanceScores.get(q.toString()).get(d1.url.toString()); 
				for (int j = i; j < docs.size(); j++) {
					Document d2 = docs.get(j);
					double rel2 = relevanceScores.get(q.toString()).get(d2.url.toString()); 
					double[] instance = extractData(q, d1, d2, rel1, rel2, idfs);
					Instance inst = new DenseInstance(1.0, instance);
					dataset.add(inst);
				}
			}
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return standardize(dataset);
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
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		features = new Instances("train_dataset", attributes, 0);
		
		Map<Query,List<Document>> queryDict = Util.loadTrainData(test_data_file);
		
		for (Entry<Query, List<Document>> entry : queryDict.entrySet()) {
			Query q = entry.getKey();
			
			int index = 0;
			testFeatures.index_map.put(q.query, new HashMap<String,Integer>());
			List<Document> docs = entry.getValue();
			
			for (int i = 0; i < docs.size(); i++) {
				Document d1 = docs.get(i);
				for (int j = i; j < docs.size(); j++) {
					Document d2 = docs.get(j);
					double[] instance = extractData(q, d1, d2, 0.0, 0.0, idfs);
					Instance inst = new DenseInstance(1.0, instance);
					features.add(inst);
					
					testFeatures.index_map.get(q.query).put(d1.url + "," + d2.url, index);
					index++;	
				}
			}
		}
		
		/* Set last attribute as target */
		features.setClassIndex(features.numAttributes() - 1);
		testFeatures.features = standardize(features);
		return testFeatures;
		
	}

	
	private double linearScore(TestFeatures tf, Classifier model, String q, String str) {
		double[] weights = ((LibSVM)model).coefficients();
		double[] x = tf.features.get(tf.index_map.get(q).get(str)).toDoubleArray();
		double score = 0.0;
		for (int i = 0; i < x.length; i++) {
			score += weights[i] * x[i];
		}
		return score;
	}
	
	@Override
	public Map<String, List<String>> testing(final TestFeatures tf,
			final Classifier model) {
		Map<String, List<String>> allRankings = new HashMap<String, List<String>>();
		for (Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
			final String q = entry.getKey();
			List<String> rankings = new ArrayList<String>();
			rankings.addAll(entry.getValue().keySet());
			if (linear) {
				Collections.sort(rankings, new Comparator<String>() {
					@Override
					public int compare(String str1, String str2) 
					{
						Double score1 = linearScore(tf, model, q, str2);
						Double score2 = linearScore(tf, model, q, str2);
						return score2.compareTo(score1);
					}	
				});
			}
			allRankings.put(q, rankings);
		}
		return allRankings;
	}

}
