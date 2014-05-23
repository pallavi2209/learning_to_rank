package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception {
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */

		Instances dataset = null;

		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_score"));
		attributes.add(new Attribute("title_score"));
		attributes.add(new Attribute("body_score"));
		attributes.add(new Attribute("header_score"));
		attributes.add(new Attribute("anchor_score"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);

		Map<Query,List<Document>> qDict = Util.loadTrainData(train_data_file);
		Map<String, Map<String, Double>> relScores = Util.loadRelData(train_rel_file);

		for (Entry<Query,List<Document>> qDocList : qDict.entrySet()) {

			Query q = qDocList.getKey();

			for (Document doc : qDocList.getValue()) {
				Double url_score = 0.0;
				Double title_score = 0.0;
				Double body_score = 0.0;
				Double header_score = 0.0;
				Double anchor_score = 0.0;
				Double relevance_score = 0.0;

				relevance_score = relScores.get(q.query).get(doc.url);
				
				Map<String, Map<String, Double>> docTermFreq = Util.getDocTermFreqs(doc, q);
				normalizeTFs(docTermFreq, doc, q);

				for (String q_word : q.words) {
					Double df = 0.0;
					if(idfs.containsKey(q_word)){
						df = idfs.get(q_word) + 1.0;
					}else{
						df = 1.0;
					}
					
					Double idf_qword = Math.log10((Util.totFiles + 1.0)/df);
					
					Map<String, Double> tfUrl = docTermFreq.get(Util.TFTYPES[0]);
					if(tfUrl.containsKey(q_word)){
						url_score += tfUrl.get(q_word)*idf_qword;

					}
					if(docTermFreq.containsKey(Util.TFTYPES[1])){
						Map<String, Double> tfTitle = docTermFreq.get(Util.TFTYPES[1]);
						if(tfTitle.containsKey(q_word)){
							title_score += tfTitle.get(q_word)*idf_qword;
						}

					}if(docTermFreq.containsKey(Util.TFTYPES[2])){
						Map<String, Double> tfBody = docTermFreq.get(Util.TFTYPES[2]);
						if(tfBody.containsKey(q_word)){
							body_score += tfBody.get(q_word)*idf_qword;
						}

					}if(docTermFreq.containsKey(Util.TFTYPES[3])){
						Map<String, Double> tfHeader = docTermFreq.get(Util.TFTYPES[3]);
						if(tfHeader.containsKey(q_word)){
							header_score += tfHeader.get(q_word)*idf_qword;
						}

					}if(docTermFreq.containsKey(Util.TFTYPES[4])){
						Map<String, Double> tfAnchor = docTermFreq.get(Util.TFTYPES[4]);
						if(tfAnchor.containsKey(q_word)){
							anchor_score += tfAnchor.get(q_word)*idf_qword;
						}

					}
				}
				/* Add data */
				double[] instance = {url_score, title_score, body_score, header_score, anchor_score, relevance_score};
				Instance inst = new DenseInstance(1.0, instance);
				dataset.add(inst);
			}

		}

		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) throws Exception {
		LinearRegression model = new LinearRegression();
		model.buildClassifier(dataset);
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) throws Exception {

		int index = 0;
		Instances dataset = null;
		TestFeatures test_features = new TestFeatures();
		test_features.index_map = new HashMap<String, Map<String, Integer>>();
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_score"));
		attributes.add(new Attribute("title_score"));
		attributes.add(new Attribute("body_score"));
		attributes.add(new Attribute("header_score"));
		attributes.add(new Attribute("anchor_score"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("test_dataset", attributes, 0);

		Map<Query,List<Document>> qDict = Util.loadTrainData(test_data_file);

		for (Entry<Query,List<Document>> qDocList : qDict.entrySet()) {

			Query q = qDocList.getKey();
			test_features.index_map.put(q.query, new HashMap<String,Integer>());

			for (Document doc : qDocList.getValue()) {
				Double url_score = 0.0;
				Double title_score = 0.0;
				Double body_score = 0.0;
				Double header_score = 0.0;
				Double anchor_score = 0.0;
				Double relevance_score = 0.0;

				test_features.index_map.get(q.query).put(doc.url, index);
				index++;	

				Map<String, Map<String, Double>> docTermFreq = Util.getDocTermFreqs(doc, q);
				normalizeTFs(docTermFreq, doc, q);
				
				for (String q_word : q.words) {
					Double df = 0.0;
					if(idfs.containsKey(q_word)){
						df = idfs.get(q_word) + 1.0;
					}else{
						df = 1.0;
					}
					Double idf_qword = Math.log10((Util.totFiles + 1.0)/df);

					Map<String, Double> tfUrl = docTermFreq.get(Util.TFTYPES[0]);
					if(tfUrl.containsKey(q_word)){
						url_score += tfUrl.get(q_word)*idf_qword;
					}
					if(docTermFreq.containsKey(Util.TFTYPES[1])){
						Map<String, Double> tfTitle = docTermFreq.get(Util.TFTYPES[1]);
						if(tfTitle.containsKey(q_word)){
							title_score += tfTitle.get(q_word)*idf_qword;
						}

					}if(docTermFreq.containsKey(Util.TFTYPES[2])){
						Map<String, Double> tfBody = docTermFreq.get(Util.TFTYPES[2]);
						if(tfBody.containsKey(q_word)){
							body_score += tfBody.get(q_word)*idf_qword;
						}

					}if(docTermFreq.containsKey(Util.TFTYPES[3])){
						Map<String, Double> tfHeader = docTermFreq.get(Util.TFTYPES[3]);
						if(tfHeader.containsKey(q_word)){
							header_score += tfHeader.get(q_word)*idf_qword;
						}

					}if(docTermFreq.containsKey(Util.TFTYPES[4])){
						Map<String, Double> tfAnchor = docTermFreq.get(Util.TFTYPES[4]);
						if(tfAnchor.containsKey(q_word)){
							anchor_score += tfAnchor.get(q_word)*idf_qword;
						}

					}
				}
				/* Add data */
				double[] instance = {url_score, title_score, body_score, header_score, anchor_score, relevance_score};
				Instance inst = new DenseInstance(1.0, instance);
				dataset.add(inst);

			}
		}

		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		test_features.features = dataset;
		return test_features;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) throws Exception {

		Map<String, List<String>> rankedResults = new HashMap<String, List<String>>();
		Instances test_dataset = tf.features;
		Map<String, Map<String, Integer>> indexMap = tf.index_map;

		for (Entry<String, Map<String, Integer>> entry : indexMap.entrySet()) {
			String query = entry.getKey();

			List<Pair<String,Double>> urlAndScores = new ArrayList<Pair<String,Double>>(entry.getValue().size());
			for (Entry<String, Integer> entryIndx : entry.getValue().entrySet()) {
				String docUrl = entryIndx.getKey();
				Integer indx = entryIndx.getValue();
				double predicted_score = model.classifyInstance(test_dataset.instance(indx));
				//to sort docs according to scores
				urlAndScores.add(new Pair<String,Double>(docUrl,predicted_score));
			}

			//sort urls for query based on scores
			Collections.sort(urlAndScores, new Comparator<Pair<String,Double>>() {
				@Override
				public int compare(Pair<String, Double> o1, Pair<String, Double> o2) 
				{
					Double score1 = o1.getSecond();
					Double score2 = o2.getSecond();
					return score2.compareTo(score1);
				}	
			});

			//put completed rankings into map
			List <String> rankedDocs = new ArrayList<String>();
			for (Pair<String,Double> urlAndScore : urlAndScores){
				rankedDocs.add(urlAndScore.getFirst());
			}

			rankedResults.put(query, rankedDocs);
		}
		return rankedResults;
	}

	double smoothingBodyLength = 500;

	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q)
	{
		Double doc_length = (double)(d.body_length + smoothingBodyLength);
		for (Entry<String,Map<String, Double>> tfTypeEntry : tfs.entrySet()) {
			String tf_type = tfTypeEntry.getKey();
			for (Entry<String, Double> tfEntry : tfTypeEntry.getValue().entrySet()) {
				String query_word = tfEntry.getKey();
				Double value = tfEntry.getValue();
				Double norm_value = value/doc_length;
				tfs.get(tf_type).put(query_word, norm_value);
			}
		}
	}

}
