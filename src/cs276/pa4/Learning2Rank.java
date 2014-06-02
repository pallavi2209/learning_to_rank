package cs276.pa4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.io.*;



import weka.classifiers.Classifier;
import weka.core.Instances;

public class Learning2Rank {

	
	public static Classifier train(String train_data_file, String train_rel_file, int task, Map<String,Double> idfs) throws Exception {
	    System.err.println("## Training with feature_file =" + train_data_file + ", rel_file = " + train_rel_file + " ... \n");
	    Classifier model = null;
	    Learner learner = null;
    
 		if (task == 1) {
			learner = new PointwiseLearner();
		} else if (task == 2) {
		  
		  double C = 1.0;
		  double gamma = 0.25;
		  // String line;
		  // BufferedReader reader = new BufferedReader(new FileReader("svmtemp.txt"));
		  // try {
		  // 	line = reader.readLine();
		  // 	C = Double.parseDouble(line);
		  // 	line = reader.readLine();
		  // 	gamma = Double.parseDouble(line);
		  // } catch (Exception e) {
		  // 	System.err.println("Error while getting C and gamma.");
		  // 	System.exit(1);
		  // }
		  boolean isLinearKernel = false;
		  learner = new PairwiseLearner(C, gamma, isLinearKernel);
		} else if (task == 3) {
			// boolean isLinearKernel = true;
			// learner = new PairwiseAddedFeatures(isLinearKernel);
			double C = 1.0;
			double gamma = 0.25;
			boolean isLinearKernel = false;
			learner = new PairwiseAddedFeatures(C, gamma, isLinearKernel);
			System.err.println("Task 3");

		} else if (task == 4) {
			
			/* 
			 * @TODO: Your code here, extra credit 
			 * */
			System.err.println("Extra credit");
			
		}
 		
		/* Step (1): construct your feature matrix here */
		Instances data = learner.extract_train_features(train_data_file, train_rel_file, idfs);
		
		/* Step (2): implement your learning algorithm here */
		model = learner.training(data);
	 		
	    return model;
	  }

	 public static Map<String, List<String>> test(String test_data_file, Classifier model, int task, Map<String,Double> idfs) throws Exception{
		 	System.err.println("## Testing with feature_file=" + test_data_file + " ... \n");
		    Map<String, List<String>> ranked_queries = new HashMap<String, List<String>>();
		    Learner learner = null;
	 		if (task == 1) {
				learner = new PointwiseLearner();
			} else if (task == 2) {
//			  boolean isLinearKernel = true;
//				learner = new PairwiseLearner(isLinearKernel);
				double C = 1.0;
				double gamma = 0.25;
				boolean isLinearKernel = false;
				learner = new PairwiseLearner(C, gamma, isLinearKernel);
			} else if (task == 3) {
//				boolean isLinearKernel = true;
//				learner = new PairwiseAddedFeatures(isLinearKernel);
				double C = 1.0;
				double gamma = 0.25;
				boolean isLinearKernel = false;
				learner = new PairwiseAddedFeatures(C, gamma, isLinearKernel);
				System.err.println("Task 3");
				
			} else if (task == 4) {
				
				/* 
				 * @TODO: Your code here, extra credit 
				 * */
				System.err.println("Extra credit");
				
			}
		 
	 		/* Step (1): construct your test feature matrix here */
	 		TestFeatures tf = learner.extract_test_features(test_data_file, idfs);
	 		
	 		/* Step (2): implement your prediction and ranking code here */
			ranked_queries = learner.testing(tf, model);
			
		    return ranked_queries;
		}
	

	/* This function output the ranking results in expected format */
	public static void writeRankedResultsToFile(Map<String,List<String>> ranked_queries, PrintStream ps) {
	    for (String query : ranked_queries.keySet()){
	      ps.println("query: " + query.toString());

	      for (String url : ranked_queries.get(query)) {
	        ps.println("  url: " + url);
	      }
	    }
	}
	

	public static void main(String[] args) throws Exception {
	    if (args.length != 4 && args.length != 5 && args.length != 7) {
	      System.err.println("Input arguments: " + Arrays.toString(args));
	      System.err.println("Usage: <train_data_file> <train_rel_file> <test_data_file> <task> [ranked_out_file]");
	      System.err.println("  ranked_out_file (optional): output results are written into the specified file. "
	          + "If not, output to stdout.");
	      return;
	    }

	    String train_data_file = args[0];
	    String train_rel_file = args[1];
	    String test_data_file = args[2];
	    int task = Integer.parseInt(args[3]);
	    String ranked_out_file = "";
	    if (args.length >= 5){
	      ranked_out_file = args[4];
	    }
	    
	    /* Populate idfs */
	    String dfFile = "df.txt";
	    Map<String,Double> idfs = null;
	    try {
	      idfs = Util.loadDFs(dfFile);
	    } catch(IOException e){
	      e.printStackTrace();
	    }

	    System.err.println("Num args: " + args.length);
	    if (args.length == 7) {
	    	try {
	    		File file = new File("svmtemp.txt");
	    		if (!file.createNewFile()) {
	    			System.err.println("File already exists.");
	    			System.exit(1);
	    		}
	    		BufferedWriter writer = new BufferedWriter(new FileWriter(file));
	    		writer.write(args[5]);
	    		writer.newLine();
	    		writer.write(args[6]);
	    		writer.newLine();
	    		writer.close();
	    	} catch (Exception e) {
	    		System.err.println("An error occured.");
	    		System.exit(1);
	    	}
	    }

	    
	    
	    /* Train & test */
	    System.err.println("### Running task" + task + "...");		
	    Classifier model = train(train_data_file, train_rel_file, task, idfs);

      /* performance on the training data */
      Map<String, List<String>> trained_ranked_queries = test(train_data_file, model, task, idfs);
      String trainOutFile="tmp.train.ranked";
      writeRankedResultsToFile(trained_ranked_queries, new PrintStream(new FileOutputStream(trainOutFile)));
      NdcgMain ndcg = new NdcgMain(train_rel_file);
      double score = ndcg.score(trainOutFile);
      System.err.println("# Trained NDCG=" + score);

      




      (new File(trainOutFile)).delete();
      
	    Map<String, List<String>> ranked_queries = test(test_data_file, model, task, idfs);
	    
	    /* Output results */
	    if(ranked_out_file.equals("")){ /* output to stdout */
	      writeRankedResultsToFile(ranked_queries, System.out);
	    } else { 						/* output to file */
	      try {
	        writeRankedResultsToFile(ranked_queries, new PrintStream(new FileOutputStream(ranked_out_file)));
	      } catch (FileNotFoundException e) {
	        e.printStackTrace();
	      }
	    }

	    ndcg = new NdcgMain("data/pa4.rel.dev");
	    score = ndcg.score(ranked_out_file);
	    System.err.println("DEV SCORE: " + score);
	    if (args.length == 7) {
	    	String filename = "C" + args[5] + "Gamma" + args[6];
	    	try {
	    		File file = new File("results/" + filename);
	    		if (!file.createNewFile()) {
	    			System.err.println("File already exists.");
	    			System.exit(1);
	    		}
	    		BufferedWriter writer = new BufferedWriter(new FileWriter(file));
	    		writer.write("C: " + args[5]);
	    		writer.newLine();
	    		writer.write("Gamma: " + args[6]);
	    		writer.newLine();
	    		writer.write("Score: " + score);
	    		writer.newLine();
	    		writer.close();
	    	} catch (Exception e) {
	    		System.err.println("An error occured.");
	    		System.exit(1);
	    	}
	    }
	}
}
