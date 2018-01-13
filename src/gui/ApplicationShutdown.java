package gui;

import java.awt.event.*;

//import coser.common.*;
//import coser.gui.dialog.common.*;

/**
 * Project: The cost-sensitive rough sets project.
 * <p>
 * Summary: Just for closing the whole project/frame.
 * <p>
 * Author: <b>Min Wang</b> wangmin80616@163.com <br>
 * Copyright: The source code and all documents are open and free. 
 */
public class ApplicationShutdown implements WindowListener, ActionListener {
	public static ApplicationShutdown applicationShutdown = new ApplicationShutdown();

	/**
	 *************************** 
	 * Empty constructor
	 *************************** 
	 */
	private ApplicationShutdown() {
	}// Of ApplicationShutdown.

	/**
	 *************************** 
	 * Shutdown the system
	 *************************** 
	 */
	public void windowClosing(WindowEvent comeInWindowEvent) {
		GUICommon.mainFrame.dispose();
		System.exit(0);
	}// Of windowClosing.

	public void windowActivated(WindowEvent comeInWindowEvent) {
	}// Of windowActivated.

	public void windowClosed(WindowEvent comeInWindowEvent) {
	}// Of windowClosed.

	public void windowDeactivated(WindowEvent comeInWindowEvent) {
	}// Of windowDeactivated.

	public void windowDeiconified(WindowEvent comeInWindowEvent) {
	}// Of windowDeiconified.

	public void windowIconified(WindowEvent comeInWindowEvent) {
	}// Of windowIconified.

	public void windowOpened(WindowEvent comeInWindowEvent) {
	}// Of windowOpened.

	/**
    *************************
    *************************
    */
	public void actionPerformed(ActionEvent ee) {
		GUICommon.mainFrame.dispose();
		System.exit(0);
	}// Of actionPerformed.

}// Of class ApplicationShutdown
