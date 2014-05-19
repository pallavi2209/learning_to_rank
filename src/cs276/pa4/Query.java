package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;

public class Query implements Comparable<Query>{
	String query;
	List<String> words; /* Words with no duplicates and all lower case */
	
	public Query(String query) {
		this.query = new String(query);
		String[] words_array = query.toLowerCase().split(" ");	
		
		
		
		// Use LinkedHashSet to remove duplicates
		words_array = (new LinkedHashSet<String>(Arrays.asList(words_array))).toArray(new String[0]);
		words = new ArrayList<String>(Arrays.asList(words_array));
	}
	
	@Override
	public int compareTo(Query arg0) {
		return this.query.compareTo(arg0.query);
	}
	
	@Override
	public String toString() {
	  return query;
	}
}
