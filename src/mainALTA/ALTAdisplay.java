package mainALTA;

import java.awt.Frame;

import gui.*;
//import coser.gui.guicommon.GUICommon;
//import coser.gui.menu.CoserMenuBar;

/**
 * The main entrance of the software.
 * 
 * @author Fan Min 2017.1.14.
 * 
 */
public class ALTAdisplay {
	/**
	 * The entrance method.
	 */
	public static void main(String args[]) {
		// A simple frame to contain dialogs.
		Frame mainFrame = new Frame();
		mainFrame
				.setTitle("Active learning through two-stage clustering. http://www.fansmale.com, wangmin80616@163.com");
		try {
			GUICommon.setFrame(mainFrame);
		} catch (Exception ee) {
			System.out.println(ee.toString());
			System.exit(0);
		}// Of try
		mainFrame.setMenuBar(DensityMenuBar.densityMenuBar);
		// ���ġ�ClusterMenuBar.clusterMenuBar����Ӷ���
		// �������ɸ���munu
		// Basic settings of the frame
		mainFrame.setSize(800, 550);
		mainFrame.setLocation(50, 50);
		mainFrame.addWindowListener(ApplicationShutdown.applicationShutdown);
		// ApplicationShutdown.applicationShutdown����������һ������
		// ���ڶ�һϵ���¼����з�Ӧ��
		// �������¼�������Ӧ��
		mainFrame.setBackground(GUICommon.MY_COLOR);
		mainFrame.setVisible(true);
	}// Of main
}
