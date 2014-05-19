package cs276.pa4;

import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {
	
	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file, Map<String,Double> idfs);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file, Map<String,Double> idfs);
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);
}
