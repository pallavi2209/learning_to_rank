package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import cs276.pa4.stemmer.EnglishSnowballStemmerFactory;


public class Util {
	

	public static final Double totFiles = 98998.0;
  public static Map<Query,List<Document>> loadTrainData (String feature_file_name) throws Exception {
    Map<Query, List<Document>> result = new HashMap<Query, List<Document>>();

    File feature_file = new File(feature_file_name);
    if (!feature_file.exists() ) {
      System.err.println("Invalid feature file name: " + feature_file_name);
      return null;
    }

    BufferedReader reader = new BufferedReader(new FileReader(feature_file));
    String line = null, anchor_text = null;
    Query query = null;
    Document doc = null;
    int numQuery=0; int numDoc=0;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")){
        query = new Query(value);
        numQuery++;
        result.put(query, new ArrayList<Document>());
      } else if (key.equals("url")) {
        doc = new Document();
        doc.url = new String(value);
        result.get(query).add(doc);
        numDoc++;
      } else if (key.equals("title")) {
        doc.title = new String(value);
      } else if (key.equals("header"))
      {
        if (doc.headers == null)
          doc.headers =  new ArrayList<String>();
        doc.headers.add(value);
      } else if (key.equals("body_hits")) {
        if (doc.body_hits == null)
          doc.body_hits = new HashMap<String, List<Integer>>();
        String[] temp = value.split(" ", 2);
        String term = temp[0].trim();
        List<Integer> positions_int;

        if (!doc.body_hits.containsKey(term))
        {
          positions_int = new ArrayList<Integer>();
          doc.body_hits.put(term, positions_int);
        } else
          positions_int = doc.body_hits.get(term);

        String[] positions = temp[1].trim().split(" ");
        for (String position : positions)
          positions_int.add(Integer.parseInt(position));

      } else if (key.equals("body_length"))
        doc.body_length = Integer.parseInt(value);
      else if (key.equals("pagerank"))
        doc.page_rank = Integer.parseInt(value);
      else if (key.equals("anchor_text")) {
        anchor_text = value;
        if (doc.anchors == null)
          doc.anchors = new HashMap<String, Integer>();
      }
      else if (key.equals("stanford_anchor_count"))
        doc.anchors.put(anchor_text, Integer.parseInt(value));      
    }

    reader.close();
    System.err.println("# Signal file " + feature_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);

    return result;
  }

  public static Map<String,Double> loadDFs(String dfFile) throws IOException {
    Map<String,Double> dfs = new HashMap<String, Double>();

    BufferedReader br = new BufferedReader(new FileReader(dfFile));
    String line;
    while((line=br.readLine())!=null){
      line = line.trim();
      if(line.equals("")) continue;
      String[] tokens = line.split("\\s+");
      dfs.put(tokens[0], Double.parseDouble(tokens[1]));
    }
    br.close();
    return dfs;
  }

  /* query -> (url -> score) */
  public static Map<String, Map<String, Double>> loadRelData(String rel_file_name) throws IOException{
    Map<String, Map<String, Double>> result = new HashMap<String, Map<String, Double>>();

    File rel_file = new File(rel_file_name);
    if (!rel_file.exists() ) {
      System.err.println("Invalid feature file name: " + rel_file_name);
      return null;
    }

    BufferedReader reader = new BufferedReader(new FileReader(rel_file));
    String line = null, query = null, url = null;
    int numQuery=0; 
    int numDoc=0;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")){
        query = value;
        result.put(query, new HashMap<String, Double>());
        numQuery++;
      } else if (key.equals("url")){
        String[] tmps = value.split(" ", 2);
        url = tmps[0].trim();
        double score = Double.parseDouble(tmps[1].trim());
        result.get(query).put(url, score);
        numDoc++;
      }
    }	
    reader.close();
    System.err.println("# Rel file " + rel_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);
    
    return result;
  }
  
	public static String scrub(String input){
		return input.toLowerCase().replaceAll("[^0-9a-z]+", " ").replace("\\s+", " ").trim();
	}
	
	public static String[] TFTYPES = { "url", "title", "body", "header", "anchor" };
	public static Double INCR = 1.0d;

	public static Map<String, Map<String, Double>> getDocTermFreqs(Document d, Query q) {
		// map from tf type -> queryWord -> score
		Map<String, Map<String, Double>> tfs = new HashMap<String, Map<String, Double>>();

		String sUrl = TFTYPES[0]; // "url"
		String docUrl = scrub(d.url);
		String[] docUrlWords = docUrl.split("\\s+");

		tfs.put(sUrl, new HashMap<String, Double>());
		for (String query_word : q.words) {
			Double count = 0.0d;
			for (int i = 0; i < docUrlWords.length; i++) {
				if (stem(query_word).equals(stem(docUrlWords[i]))) {
					tfs.get(sUrl).put(query_word, count += INCR);
				}
			}
		}

		String sTitle = TFTYPES[1]; // title
		String docTitle = d.title;
		if (docTitle != null) {
			String[] docTitleWords = docTitle.toLowerCase().split("\\s+");
			tfs.put(sTitle, new HashMap<String, Double>());
			for (String query_word : q.words) {
				Double count = 0.0d;
				for (int i = 0; i < docTitleWords.length; i++) {
					if (stem(query_word).equals(stem(docTitleWords[i]))) {
						tfs.get(sTitle).put(query_word, count += INCR);
					}
				}
			}
		}

		String sBody = TFTYPES[2];// body
		Map<String, List<Integer>> docBodyHits = d.body_hits;
		if (docBodyHits != null) {
			tfs.put(sBody, new HashMap<String, Double>());
			for (String query_word : q.words) {
				if (docBodyHits.containsKey(query_word)) {
					Double count = Double.valueOf((double) docBodyHits.get(
							query_word).size());
					tfs.get(sBody).put(query_word, count);
				}
			}
		}


		String sHeader = TFTYPES[3]; // header
		List<String> docHeaders = d.headers;
		if (docHeaders != null) {
			tfs.put(sHeader, new HashMap<String, Double>());

			for (String query_word : q.words) {
				Double count = 0.0d;

				for (String string : docHeaders) {
					String[] docHeaderWords = string.toLowerCase().split("\\s+");

					for (int j = 0; j < docHeaderWords.length; j++) {
						if (stem(query_word).equals(stem(docHeaderWords[j]))) {
							tfs.get(sHeader).put(query_word, count += INCR);
						}
					}
				}
			}
		}

		String sAnchor = TFTYPES[4];
		Map<String, Integer> docAnchors = d.anchors;
		if (docAnchors != null) {
			tfs.put(sAnchor, new HashMap<String, Double>());

			for (String query_word : q.words) {
				Double count = 0.0d;

				for (Entry<String, Integer> anchorEntry : docAnchors.entrySet()) {
					String[] anchorWords = anchorEntry.getKey().toLowerCase()
							.split("\\s+");
					Double countThisAnchor = Double
							.valueOf((double) anchorEntry.getValue());

					for (int j = 0; j < anchorWords.length; j++) {
						if (stem(query_word).equals(stem(anchorWords[j]))) {
							if (tfs.get(sAnchor).containsKey(query_word)) {
								Double prevCount = tfs.get(sAnchor).get(
										query_word);
								count = prevCount + countThisAnchor;
							} else {
								count = countThisAnchor;
							}
							tfs.get(sAnchor).put(query_word, count);
						}
					}

				}
			}
		}
		return tfs;
	}

	public static String stem(String word){
		String stem_word = word;
		try {
			stem_word = EnglishSnowballStemmerFactory.getInstance().process(word);

		} catch (Exception e) {
			e.printStackTrace();
		}
		return stem_word;
	}
	
  public static void main(String[] args) {
    try {
      System.out.print(loadRelData(args[0]));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
