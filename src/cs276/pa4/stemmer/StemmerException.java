package cs276.pa4.stemmer;

public class StemmerException extends Exception {

	private static final long serialVersionUID = 1L;

	private String message;

	private Object[] params;

	/**
	 * @param e
	 */
	public StemmerException(Exception e) {
		e.printStackTrace();
		setMessage(e.getMessage());
	}

	/**
	 * @param e
	 * @param message
	 */
	public StemmerException(Exception e, String message) {
		e.printStackTrace();
		setMessage(message);
	}

	/**
	 * @param message
	 */
	public StemmerException(String message) {
		// ensures this message is at least defined and logged somewhere
		message = (null == message || 0 == message.length()) ? "Undefined Error Message" : message;
		setMessage(message);
	}

	/**
	 * @param message
	 * @param params
	 */
	public StemmerException(String message, Object... params) {
		this(String.format(message, params));
		this.params = params;
	}

	/**
	 * @return the message
	 */
	@Override
	public String getMessage() {
		return message;
	}

	/**
	 * 
	 * @return params the exception parameters
	 */
	public Object[] getParams() {
		return params;
	}

	/**
	 * @param message the message to set
	 */
	public void setMessage(String message) {
		this.message = message;
	}

	/**
	 * 
	 * @param params
	 */
	public void setParams(Object[] params) {
		this.params = params;
	}
}
