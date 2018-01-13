package algorithms;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

import weka.core.Instances;

public class ALTA extends Instances {
	int[] CSALLables;
	int[] cl;
	int[] Lables;
	double TotalCost;
	double Accuracy;
	int[] classLabels;
	int numTeach;
	int numVote;
	int k;
	int n;
	int[][] blockInformationActive;
	double[][] centerskMeans;
	int[] predictedLablesKmeans;
	int[] blockSizes;
	int[] descendantIndices;
	int[][] blockInformation;
	boolean[] alreadyLabled;
	int[] representativePoints;
	double[] rho;
	int[] ordrho;
	double[] delta;
	int[] master;
	double[] priority;
	int[] ordpriority;
	double maximalDistance;
	int[] centersDensity;
	int[] predictedLabels;
	int[] clusterIndices;

	/**
	 ********************************** 
	 * Read from a reader
	 ********************************** 
	 */
	public ALTA(Reader paraReader) throws IOException, Exception {
		super(paraReader);

		classLabels = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			classLabels[i] = (int) instance(i).value(numAttributes() - 1);

		}// Of for i

	}// Of the first constructor

	/**
	 ********************************** 
	 * Stage I. 2-round K-means clustering.
	 ********************************** 
	 */

	/**
	 *************** 
	 * Step 1. Select the centers(uniform).
	 **************** 
	 */
	public void SelectCenters() {
		int[] tempIndex = generateUniformInt();

		for (int i = 0; i < tempIndex.length; i++) {
			for (int j = 0; j < numAttributes() - 1; j++) {
				centerskMeans[i][j] = instance(tempIndex[i]).value(j);
			}// of for j
		}// of for i
	}// of SelcetCenters

	/**
	 *************** 
	 * Step 2. Compute the distance.
	 **************** 
	 */
	public double computeDistance(int paraI, double[] paraArray) {

		double tempDistance = 0;
		for (int i = 0; i < numAttributes() - 1; i++) {
			tempDistance += Math.abs(instance(paraI).value(i) - paraArray[i]);
		}// Of for i

		return tempDistance;
	}// Of distance

	/**
	 *************** 
	 * Step 3. Cluster using centers.
	 **************** 
	 */

	public void clusterUsingCenters() {
		predictedLablesKmeans = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			int tempIndex = 0;
			double tempDistance = Double.MAX_VALUE;

			for (int j = 0; j < centerskMeans.length; j++) {
				if (computeDistance(i, centerskMeans[j]) < tempDistance) {
					tempDistance = computeDistance(i, centerskMeans[j]);
					tempIndex = j;
				}// Of if
			}// Of for j

			if (predictedLablesKmeans[i] != tempIndex) {

				predictedLablesKmeans[i] = tempIndex;
			}// Of if
		}// Of for i
	}// Of clusterUsingCenters

	/**
	 *************** 
	 * Step 4. Compute new centers using the mean value of each block.
	 **************** 
	 */
	public void meanAsCenters() {
		// Initialize
		blockSizes = new int[k];
		for (int i = 0; i < centerskMeans.length; i++) {
			blockSizes[i] = 0;
			for (int j = 0; j < centerskMeans[i].length; j++) {
				centerskMeans[i][j] = 0;
			}// Of for j
		}// Of for i

		// Scan all instances and sum
		for (int i = 0; i < numInstances(); i++) {
			blockSizes[predictedLablesKmeans[i]]++;
			for (int j = 0; j < numAttributes() - 1; j++) {
				centerskMeans[predictedLablesKmeans[i]][j] += instance(i)
						.value(j);
			}// Of for j
		}// Of for i

		// Divide
		for (int i = 0; i < centerskMeans.length; i++) {
			for (int j = 0; j < centerskMeans[i].length; j++) {
				centerskMeans[i][j] /= blockSizes[i];
			}// Of for j
		}// Of for i
	}// Of meanAsCenters

	/**
	 *************** 
	 * Step 5. 2-round K-Means clustering.
	 **************** 
	 */
	public void clusterKMeans() {
		// Initialize
		k = (int) Math.ceil(Math.sqrt(numInstances()));
		centerskMeans = new double[k][numAttributes() - 1];
		int tempNumber = 0;

		// Select centers
		SelectCenters();

		// Cluster and mean
		while (tempNumber < 2) {
			// Cluster

			clusterUsingCenters();

			// Mean
			meanAsCenters();

			tempNumber++;

		}// Of while
	}// Of cluster

	/**
	 ********************************** 
	 * Stage II. density clustering.
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Step 1. Compute block information according to the kMeans cluster.
	 ********************************** 
	 */
	public void computeBlockInformation() {
		int tempBlocks = centerskMeans.length;
		blockInformation = new int[centerskMeans.length][];

		for (int i = 0; i < tempBlocks; i++) {
			// Scan to see how many elements
			int tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (predictedLablesKmeans[j] == i) {
					tempElements++;
				}// Of if
			}// Of for k

			// Copy to the list
			blockInformation[i] = new int[tempElements];
			tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (predictedLablesKmeans[j] == i) {
					blockInformation[i][tempElements] = j;
					tempElements++;
				}// Of if
			}// Of for k
		}// Of for i
	}// Of computeBlockInformation

	/**
	 ********************************** 
	 * Step 2. Compute representative, generate the new data set.
	 ********************************** 
	 */
	public void computeRepresentative() {
		representativePoints = new int[k];
		double tempValue;
		for (int i = 0; i < blockInformation.length; i++) {
			tempValue = 1000;

			for (int j = 0; j < blockInformation[i].length; j++) {
				if (computeDistance(j, centerskMeans[i]) < tempValue) {
					tempValue = computeDistance(j, centerskMeans[i]);
					representativePoints[i] = blockInformation[i][j];
				}// of if
			}// of for j
		}// of for i
	}// of computeRepresentative

	/**
	 ********************************** 
	 * Step 2. Compute rho
	 ********************************** 
	 */
	public void computeRho() {
		rho = new double[k];

		for (int i = 0; i < blockInformation.length; i++) {
			int tempElements = 0;
			for (int j = 0; j < blockInformation[i].length; j++) {
				tempElements++;
				rho[i] = tempElements;
			}// of for j

		}// of for i

	}// Of computeRho

	/**
	 ********************************** 
	 * Step 3. Compute delta
	 ********************************** 
	 */
	public void computeDelta() {
		delta = new double[k];
		master = new int[k];
		ordrho = new int[k];
		ordrho = mergeSortToIndices(rho);

		delta[ordrho[0]] = Double.MAX_VALUE;

		for (int i = 1; i < k; i++) {
			delta[ordrho[i]] = Double.MAX_VALUE;
			for (int j = 0; j <= i - 1; j++) {
				if (manhattan(representativePoints[ordrho[i]],
						representativePoints[ordrho[j]]) < delta[ordrho[i]]) {
					delta[ordrho[i]] = manhattan(
							representativePoints[ordrho[i]],
							representativePoints[ordrho[j]]);
					master[ordrho[i]] = ordrho[j];
				} // of if
			}// of for j
		}// of for i

	}// Of computeDelta

	/**
	 ********************************** 
	 * Step 3. Compute priority.
	 ********************************** 
	 */
	public void computePriority() {
		priority = new double[k];
		for (int i = 0; i < k; i++) {
			priority[i] = rho[i] * delta[i];
		}// Of for i
	}// Of computePriority

	/**
	 ********************************** 
	 * Step 4. Compute centers
	 ********************************** 
	 */
	public void computeCentersDensity(int paraK) {
		n = paraK;
		centersDensity = new int[n];
		ordpriority = new int[k];

		computePriority();

		ordpriority = mergeSortToIndices(priority);

		for (int i = 0; i < n; i++) {
			centersDensity[i] = ordpriority[i];
		}// of for i

	}// Of computeCenters

	/**
	 ********************************** 
	 * Step 5. Cluster according to the centers
	 ********************************** 
	 */
	public void clusterDensity() {

		predictedLabels = new int[numInstances()];
		cl = new int[k];
		clusterIndices = new int[k];

		for (int i = 0; i < k; i++) {
			cl[i] = -1;
		}// of for i

		for (int i = 0; i < n; i++) {

			cl[centersDensity[i]] = i;

		}// of for i

		for (int i = 0; i < k; i++) {

			if (cl[ordrho[i]] == -1) {

				cl[ordrho[i]] = cl[master[ordrho[i]]];

			}// of if
		}// of for i

		for (int i = 0; i < k; i++) {
			clusterIndices[i] = centersDensity[cl[i]];
		}

		for (int i = 0; i < blockInformation.length; i++) {

			for (int j = 0; j < blockInformation[i].length; j++) {

				predictedLabels[blockInformation[i][j]] = cl[i];

			}// of if
		}// of for i

	}// Of clusterWithCenters

	/**
	 ********************************** 
	 * Step 1. Compute block information according to the TSD cluster.
	 ********************************** 
	 */
	public void computeBlockInformationActive(int paraK) {

		blockInformationActive = new int[paraK][];

		for (int i = 0; i < paraK; i++) {
			// Scan to see how many elements
			int tempElements = 0;
			for (int j = 0; j < k; j++) {
				if (cl[j] == i) {
					tempElements++;
				}// Of if
			}// Of for k

			// Copy to the list
			blockInformationActive[i] = new int[tempElements];
			tempElements = 0;
			for (int j = 0; j < k; j++) {
				if (cl[j] == i) {
					blockInformationActive[i][tempElements] = j;
					tempElements++;
				}// Of if
			}// Of for k

		}// Of for i

	}// Of computeBlockInformation

	public void activeLearning(int paraK, int paraTeachEachBlock, int paraTeach) {

		int tempBlocks;
		tempBlocks = paraK;
		CSALLables = new int[numInstances()];
		Lables = new int[numInstances()];
		descendantIndices = new int[k];

		computeRho();
		computeDelta();
		computePriority();
		numTeach = 0;

		descendantIndices = mergeSortToIndices(priority);
		alreadyLabled = new boolean[numInstances()];

		while (true) {

			computeCentersDensity(tempBlocks);

			clusterDensity();

			computeBlockInformationActive(tempBlocks);

			// Step 1 �ж���Щ���Ѿ�������
			boolean[] tempBlockProcessed = new boolean[tempBlocks];
			int tempUnProcessedBlocks = 0;
			for (int i = 0; i < blockInformationActive.length; i++) {
				tempBlockProcessed[i] = true;
				for (int j = 0; j < blockInformationActive[i].length; j++) {

					if (!alreadyLabled[representativePoints[blockInformationActive[i][j]]]) {
						tempBlockProcessed[i] = false;
						tempUnProcessedBlocks++;
						break;
					}// of if
				}// of for j
			}// of for i

			// Step 2.3 �����ǩ������ѧϰ
			// �����ǩ�������¼������

			for (int i = 0; i < blockInformationActive.length; i++) {
				// Step 2.3.1 ����ÿ��Ѿ��������꣬��ֱ���˳�������Ҫ����

				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				// Step 2.3.2 ���ĳһ��̫С������δ�ܱ��ʵ������С��ÿ�鹺����������ȫ������

				if (blockInformationActive[i].length < paraTeachEachBlock) {

					for (int j = 0; j < blockInformationActive[i].length; j++) {
						if (!alreadyLabled[representativePoints[blockInformationActive[i][j]]]) {
							if (numTeach >= paraTeach) {
								break;
							}// of if
							CSALLables[representativePoints[blockInformationActive[i][j]]] = (int) instance(
									representativePoints[blockInformationActive[i][j]])
									.classValue();
							alreadyLabled[representativePoints[blockInformationActive[i][j]]] = true;
							numTeach++;

						}// of if
					}// of for j
				}// of if

				// Step 2.3.3 ʣ�µĿ飬�������ʵ��������ÿ���ܹ���ı�ǩ��
				double[] tempPriority = new double[blockInformationActive[i].length];
				int[] ordPriority = new int[blockInformationActive[i].length];

				int tempIndex = 0;
				for (int j = 0; j < k; j++) {
					if (clusterIndices[descendantIndices[j]] == centersDensity[i]) {
						ordPriority[tempIndex] = descendantIndices[j];
						// tempPriority[tempIndex] = priority[j];
						tempIndex++;
					}// of if
				}// of for j

				int tempNumTeach = 0;
				for (int j = 0; j < blockInformationActive[i].length; j++) {
					if (alreadyLabled[representativePoints[ordPriority[j]]]) {
						continue;
					}// of if
					if (numTeach >= paraTeach) {
						break;
					}// of if
					CSALLables[representativePoints[ordPriority[j]]] = (int) instance(
							representativePoints[ordPriority[j]]).classValue();
					alreadyLabled[representativePoints[ordPriority[j]]] = true;
					numTeach++;

					tempNumTeach++;

					if (tempNumTeach >= paraTeachEachBlock) {
						break;
					}// of if

				}// of for j

			} // of for i

			// Step 2 ��ʣ�µ�ʵ������Ԥ�⣬����

			boolean tempPure = true;

			for (int i = 0; i < blockInformationActive.length; i++) {

				// Step 2.4.1 �жϸÿ��Ƿ��Ѿ�����
				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				// Step 2.4.2 �ж�ĳ���Ƿ�Ϊ���Ŀ�

				boolean tempFirstLable = true;
				// ����һ����ǣ��ж�ǰ������ʵ���ı���Ƿ�һ��
				int tempCurrentInstance;
				int tempLable = 0;

				for (int j = 0; j < blockInformationActive[i].length; j++) {
					tempCurrentInstance = representativePoints[blockInformationActive[i][j]];
					if (alreadyLabled[tempCurrentInstance]) {

						if (tempFirstLable) {
							tempLable = (int) instance(tempCurrentInstance)
									.classValue();
							tempFirstLable = false;
						} else {
							if (tempLable != (int) instance(tempCurrentInstance)
									.classValue()) {
								tempPure = false;
								break;
							}// of if
						} // of if
					}// of if
				}// of for j

				// Step 2.4.3 ����ÿ��Ǵ��ģ�ֱ�ӷ�������ʣ�µ�ʵ���������ͬ�ı�ǩ
				if (tempPure) {
					for (int j = 0; j < blockInformationActive[i].length; j++) {
						if (!alreadyLabled[representativePoints[blockInformationActive[i][j]]]) {
							CSALLables[representativePoints[blockInformationActive[i][j]]] = tempLable;
							alreadyLabled[representativePoints[blockInformationActive[i][j]]] = true;

						}// of if
					}// of for j
				}// of if
			}// of for i

			// Step 2.4.4 ����ÿ鲻�Ǵ��ģ������¾��࣬�������ѳɸ�С�Ŀ�

			tempBlocks++;
			// Step 2.5.1 ������еĿ鶼�Ѿ����������˳�
			if (tempUnProcessedBlocks == 0) {
				break;
			}// of if
			if (numTeach >= paraTeach) {
				break;
			}// of if
		}// of while

		int max = getMax(CSALLables);
		computeCentersDensity(max + 1);
		clusterDensity();

		computeBlockInformationActive(max + 1);
		// Step 3.2 ͶƱ

		int[][] vote = new int[max + 1][max + 1];
		int voteIndex = -1;

		for (int i = 0; i < blockInformationActive.length; i++) {
			// Step 3.2.1 ͳ�Ʊ�ǩ
			for (int j = 0; j < blockInformationActive[i].length; j++) {
				for (int k = 0; k <= max; k++) {
					if (CSALLables[blockInformationActive[i][j]] == k) {
						vote[i][k]++;
					}// of if
				}// of for k
			}// of for j

			// Step 3.2.2 ����ÿһ����ʵ�����ķ���
			voteIndex = getMaxIndex(vote[i]);

			// Step 5.2.3 ����һ��������δ�����ǻ��������ı��

			for (int j = 0; j < blockInformationActive[i].length; j++) {
				if (CSALLables[blockInformationActive[i][j]] == -1) {
					CSALLables[blockInformationActive[i][j]] = voteIndex;
					numVote++;
				}// of if
			}// of for j
		}// of for i

		for (int i = 0; i < blockInformation.length; i++) {
			for (int j = 0; j < blockInformation[i].length; j++) {
				CSALLables[blockInformation[i][j]] = CSALLables[representativePoints[i]];
			}// of if
		}// of for i
	}// of activeLearning

	/**
	 ********************************** 
	 * ������ܴ��ۣ�����������ͬ��CSAL
	 ********************************** 
	 */

	public double getPrediction() {
		double tempIncorrect = 0;
		TotalCost = 0;
		Accuracy = 0;
		for (int i = 0; i < numInstances(); i++) {
			if (CSALLables[i] != (int) instance(i).classValue()) {
				tempIncorrect++;
			}// of if
		}// of for i

		Accuracy = (numInstances() - tempIncorrect - numTeach)
				/ (numInstances() - numTeach);

		return Accuracy;
		
	}// of getPredictionAccuracy

	/**
	 ********************************** 
	 * �����ֵ���ڵ�λ��.
	 ********************************** 
	 */
	public int getMaxIndex(int[] paraArray) {
		int maxIndex = 0;
		int tempIndex = 0;
		int max = paraArray[0];

		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] > max) {
				max = paraArray[i];
				tempIndex = i;
			}// of if
		}// of for i
		maxIndex = tempIndex;
		return maxIndex;
	}// of getMaxIndex

	/**
	 ********************************** 
	 * Other functions.
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Manhattan distance.
	 ********************************** 
	 */
	public double manhattan(int paraI, int paraJ) {
		double tempDistance = 0;

		for (int i = 0; i < numAttributes(); i++) {
			tempDistance += Math.abs(instance(paraI).value(i)
					- instance(paraJ).value(i));
		}// of for i

		return tempDistance;
	}// of manhattan

	/**
	 ********************************** 
	 * Generate a random sequence of [0, n - 1].
	 * 
	 * @author Hengru Zhang, Revised by Fan Min 2013/12/24
	 * 
	 * @param paraLength
	 *            the length of the sequence
	 * @return an array of non-repeat random numbers in [0, paraLength - 1].
	 ********************************** 
	 */
	public static int[] generateRandomSequence(int paraLength) {
		Random random = new Random();
		// Initialize
		int[] tempResultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			tempResultArray[i] = i;
		}// Of for i

		// Swap some elements
		int tempFirstIndex, tempSecondIndex, tempValue;
		for (int i = 0; i < paraLength / 2; i++) {
			tempFirstIndex = random.nextInt(paraLength);
			tempSecondIndex = random.nextInt(paraLength);

			// Really swap elements in these two indices
			tempValue = tempResultArray[tempFirstIndex];
			tempResultArray[tempFirstIndex] = tempResultArray[tempSecondIndex];
			tempResultArray[tempSecondIndex] = tempValue;
		}// Of for i

		return tempResultArray;
	}// Of generateRandomSequence

	/**
	 ********************************** 
	 * Generate a random sequence of [0, n - 1].
	 * 
	 * @author Hengru Zhang, Revised by Fan Min 2013/12/24
	 * 
	 * @param paraLength
	 *            the length of the sequence
	 * @return an array of non-repeat random numbers in [0, paraLength - 1].
	 ********************************** 
	 */
	public int[] generateUniformInt() {
		// Initialize
		int[] tempResultArray = new int[(int) Math.ceil(Math
				.sqrt(numInstances())) - 1];

		for (int i = 0; i < Math.ceil(Math.sqrt(numInstances())) - 1; i++) {

			tempResultArray[i] = (int) (i * Math
					.ceil(Math.sqrt(numInstances())));

		}// of for i

		return tempResultArray;
	}// Of generateRandomSequence

	/**
	 *************** 
	 * Compute the distance.
	 **************** 
	 */
	public double distanceArray(double[] paraArrayFirst,
			double[] paraArraySecond) {

		double tempDistance = 0;
		for (int i = 0; i < numAttributes() - 1; i++) {
			tempDistance += Math.abs(paraArrayFirst[i] - paraArraySecond[i]);
		}// Of for i

		return tempDistance;
	}// Of distance

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged.<br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2].<br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].
	 * 
	 * @author Fan Min 2016/09/09
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */

	public static  int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];//����ά�Ƚ����洢����tempIndex����
		
		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i
			// System.out.println("Initialize, resultMatrix = " +
			// Arrays.deepToString(resultMatrix));

		// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;
		while (tempCurrentLength < tempLength) {
			// System.out.println("tempCurrentLength = " + tempCurrentLength);
			// Divide into a number of groups
			// Here the boundary is adaptive to array length not equal to 2^k.
			// ceil������ȡ������
			
			for (int i = 0; i < Math.ceil(tempLength + 0.0 / tempCurrentLength) / 2; i++) {//��λ����һ��
				// Boundaries of the group
				tempFirstStart = i * tempCurrentLength * 2;
				//tempSecondStart��λ�ڶ��鿪ʼ��λ��index
				tempSecondStart = tempFirstStart + tempCurrentLength;//���������ж��Ƿ������һС�飬������ʼ���Ĺ���
//				if (tempSecondStart >= tempLength) {
//					
//					break;
//				} // Of if
				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {  //�������һС�顣�����������峤�ȣ���tempSecondEnd��λ���������
					tempSecondEnd = tempLength - 1;
				} // Of if
//					 System.out.println("tempFirstStart = " + tempFirstStart +
//					 ", tempSecondStart = " + tempSecondStart
//					 + ", tempSecondEnd = " + tempSecondEnd);

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;
				// System.out.println("Before merge");
				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;						
					} // Of for j
				break;
			} // Of if
				
				while ((tempFirstIndex <= tempSecondStart - 1) && (tempSecondIndex <= tempSecondEnd)) {//������ʼ������Ĺ���
					
					if (paraArray[resultMatrix[tempIndex % 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex
							% 2][tempSecondIndex]]) {
						//System.out.println("tempIndex + 1) % 2"+tempIndex);
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempFirstIndex];
						int a =(tempIndex + 1) % 2;
						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempSecondIndex];
						int b =(tempIndex + 1) % 2;
						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;
				   
				} // Of while
					// System.out.println("After compared merge");
				// Remaining part
				// System.out.println("Copying the remaining part");
				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];
					tempCurrentIndex++;
					
				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];					
					tempCurrentIndex++;
				} // Of for j
				//paraArray=resultMatrix[0];
					// System.out.println("After copying remaining part");
				// System.out.println("Round " + tempIndex + ", resultMatrix = "
				// + Arrays.deepToString(resultMatrix));
			} // Of for i
				// System.out.println("Round " + tempIndex + ", resultMatrix = "
				// + Arrays.deepToString(resultMatrix));

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices
	
	/**
	 ********************************** 
	 * Compute the maximal value.
	 ********************************** 
	 */

	public int maximal(int[] paraArray) {
		int tempMaximal = Integer.MIN_VALUE;
		for (int i = 0; i < paraArray.length; i++) {
			if (tempMaximal < paraArray[i]) {
				tempMaximal = paraArray[i];
			}// Of if
		}// Of for i

		return tempMaximal;
	}// Of maximal

	/**
	 ********************************** 
	 * �����ֵ.
	 ********************************** 
	 */
	public int getMax(int[] paraArray) {
		int max = paraArray[0];
		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] > max) {
				max = paraArray[i];
			}// of if
		}// of for i

		return max;
	}// of getMax

	/**
	 ********************************** 
	 * Compute the maximal distance.
	 ********************************** 
	 */
	public double computeMaximalDistance(double[][] paraArray) {
		maximalDistance = 0;
		double tempDistance;
		for (int i = 0; i < paraArray.length; i++) {
			for (int j = 0; j < paraArray[i].length; j++) {
				tempDistance = distanceArray(paraArray[i], paraArray[j]);
				if (maximalDistance < tempDistance) {
					maximalDistance = tempDistance;
				}// Of if
			}// Of for j
		}// Of for i

		return maximalDistance;
	}// Of setDistanceMeasure

	/**
	 ******************* 
	 * ALTA test.
	 ******************* 
	 */
	public static void ALTATest() {
		String arffFilename = "D:/data/ionosphere.arff";

		try {
			long startread = System.currentTimeMillis();
			FileReader fileReader = new FileReader(arffFilename);
			ALTA tempData = new ALTA(fileReader);
			fileReader.close();
			tempData.setClassIndex(tempData.numAttributes() - 1);
			long endread = System.currentTimeMillis();
			System.out.println("��ȡ�ļ�����ʱ��" + (endread - startread) + "����!");
			long startkmeans = System.currentTimeMillis();

			tempData.clusterKMeans();
			long endkmeans = System.currentTimeMillis();
			System.out.println("����kmeans����ʱ��" + (endkmeans - startkmeans)
					+ "����!");
			tempData.computeBlockInformation();
			tempData.computeRepresentative();
			tempData.activeLearning(2, 2, 18);
			tempData.getPrediction();
			System.out.println("Accuracy" + tempData.Accuracy);
			System.out.println("numTeach" + tempData.numTeach);

		} catch (Exception ee) {
			System.out.println("Error occurred while trying to read \'"
					+ arffFilename + "\' in TSDTest().\r\n" + ee);
		}// Of try
	}// Of densityTest

	public static void main(String[] args) {

		long start = System.currentTimeMillis();
		ALTATest();

		long end = System.currentTimeMillis();
		System.out.println("�ܻ���ʱ��" + (end - start) + "����!");

	}// Of main

}// Of TSD

