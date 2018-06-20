/*
 *    SAMkNN.java
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers.lazy;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.clusterers.kmeanspm.CoresetKMeans;
import moa.core.Measurement;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.ArrayList;

/**
 * Self Adjusting Memory (SAM) coupled with the k Nearest Neighbor classifier (kNN) .<p>
 * <p>
 * Valid options are:<p>
 * <p>
 * -k number of neighbours <br> -w max instances <br> -m minimum number of instances in the STM <br> -p LTM size relative to max instances <br> -r Recalculation of the STM error <br>
 *
 * @author Viktor Losing (vlosing@techfak.uni-bielefeld.de)
 *         Paper:
 *         "KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift"
 *         Viktor Losing, Barbara Hammer and Heiko Wersing
 *         http://ieeexplore.ieee.org/document/7837853
 *         PDF can be found at https://pub.uni-bielefeld.de/download/2907622/2907623
 *         BibTex:
 *         "@INPROCEEDINGS{7837853,
 *         author={V. Losing and B. Hammer and H. Wersing},
 *         booktitle={2016 IEEE 16th International Conference on Data Mining (ICDM)},
 *         title={KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift},
 *         year={2016},
 *         pages={291-300},
 *         keywords={data mining;optimisation;pattern classification;Big Data;Internet of Things;KNN classifier;SAM-kNN robustness;data mining;k nearest neighbor algorithm;metaparameter optimization;nonstationary data streams;performance evaluation;self adjusting memory model;Adaptation models;Benchmark testing;Biological system modeling;Data mining;Heuristic algorithms;Prediction algorithms;Predictive models;Data streams;concept drift;data mining;kNN},
 *         doi={10.1109/ICDM.2016.0040},
 *         month={Dec}
 *         }"
 */
public class SAMkNNFS2 extends AbstractClassifier {
    private static final long serialVersionUID = 1L;
    private static final String rEuclidean = "Euclidean";
    private static final String rManhatten = "Manhatten";
    private static final String rChebychev = "Chebychev";

    public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 5, 1, Integer.MAX_VALUE);

    public FlagOption uniformWeightedOption = new FlagOption("uniformWeighted", 'u',
            "uniformWeighted");

    public FlagOption normalizeDistancesOption = new FlagOption("normalizeDistances", 'n',
            "normalizeDistances");

    public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to store", 5000, 1, Integer.MAX_VALUE);
    public IntOption minSTMSizeOption = new IntOption("minSTMSize", 'm', "The minimum number of instances in the STM", 50, 1, Integer.MAX_VALUE);

    public FloatOption relativeLTMSizeOption = new FloatOption(
            "relativeLTMSize",
            'p',
            "The allowed LTM size relative to the total limit.",
            0.4, 0.0, 1.0);

    public FlagOption recalculateSTMErrorOption = new FlagOption("recalculateError", 'r',
            "Recalculates the error rate of the STM for size adaption (Costly operation). Otherwise, an approximation is used.");

    public MultiChoiceOption distanceMetricOption = new MultiChoiceOption(
            "distanceMetric", 'a', "Grace period.", new String[]{
            rEuclidean, rManhatten, rChebychev}, new String[]{
            "Euclidean",
            "Manhatten",
            "Chebychev"}, 0);

    private int maxClassValue = 0;
    private ADWIN adwin;

    @Override
    public String getPurposeString() {
        return "SAMkNN: special.";
    }

    private ExecutorService executor;
    private Instances stm;
    private Instances ltm;
    private int maxLTMSize;
    private int maxSTMSize;
    private List<Integer> stmHistory;
    private List<Integer> ltmHistory;
    private List<Integer> cmHistory;
    private float[][] distMSTM;
    private int distMStartIdx;

    STMDistanceMatrix STMDistMatrix;
    private List<Integer> stmMasterIndices;

    private float[] lastVotedInstanceDistancesSTM;
    private float[] lastVotedInstanceDistancesLTM;
    private Instance lastVotedInstance;

    private int trainStepCount;
    private Map<Integer, List<Integer>> predictionHistories;
    private int[] listAttributes;
    private int[] numericAttributes;
    private int[] nominalAttributes;
    public float accCurrentConcept;
    protected int numAttributes;
    private long runTimeMeasurement;
    private long runTimeMeasurementTrain;
    private long runTimeMeasurementVotes;


    /**
     * Index in ranges for MIN.
     */
    public static final int R_MIN = 0;

    /**
     * Index in ranges for MAX.
     */

    public static final int R_MAX = 1;

    /**
     * Index in ranges for WIDTH.
     */
    public static final int R_WIDTH = 2;

    /**
     * The range of the attributes.
     */
    protected double[][] m_Ranges;


    protected void init() {
        this.maxLTMSize = (int) (relativeLTMSizeOption.getValue() * limitOption.getValue());
        this.maxSTMSize = limitOption.getValue() - this.maxLTMSize;
        this.stmHistory = new ArrayList<>();
        this.ltmHistory = new ArrayList<>();
        this.cmHistory = new ArrayList<>();
        //store calculated STM distances in a matrix to avoid recalculation, are reused in the STM adaption phase
        this.distMSTM = new float[limitOption.getValue() + 1][limitOption.getValue() + 1];
        this.distMStartIdx = 0;
        this.predictionHistories = new HashMap<>();
        this.executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    @Override
    public void afterLearning() {
        this.stm = null;
        this.ltm = null;
        this.stmHistory = null;
        this.ltmHistory = null;
        this.cmHistory = null;
        this.distMSTM = null;
        this.predictionHistories = null;
        this.listAttributes = null;

        this.numericAttributes = null;
        this.nominalAttributes = null;
        this.STMDistMatrix = null;
        this.stmMasterIndices = null;

        if (executor != null) {
            executor.shutdown();
        }
        System.out.println("train " + runTimeMeasurementTrain/1000000000.);
        System.out.println("votes " + runTimeMeasurementVotes/1000000000.);
        System.out.println(runTimeMeasurement/1000000000.);

    }

    private void initRangesFirst(Instance inst) {
        int numAtt = inst.numAttributes() -1;
        m_Ranges = new double[numAtt][3];
        for (int j = 0; j < numAtt; j++) {
            double value = inst.value(j);
            m_Ranges[j][R_MIN] = value;
            m_Ranges[j][R_MAX] = value;
            m_Ranges[j][R_WIDTH] = 0.0;
        }
    }
    public void setMasterDistMSTM(STMDistanceMatrix distMatrix){
        stmMasterIndices = new ArrayList<>();
        this.STMDistMatrix = distMatrix;
    }
    public void addSTMMasterIdx(int idx){
        //System.out.println("add idx" + idx);
        stmMasterIndices.add(idx);
    }

    public void updateRanges(Instance inst) {
        for (int j = 0; j < inst.numAttributes()-1; j++) {
            double value = inst.value(j);
            if (value < m_Ranges[j][R_MIN]) {
                m_Ranges[j][R_MIN] = value;
                m_Ranges[j][R_WIDTH] = m_Ranges[j][R_MAX] - m_Ranges[j][R_MIN];
            } else {
                if (value > m_Ranges[j][R_MAX]) {
                    m_Ranges[j][R_MAX] = value;
                    m_Ranges[j][R_WIDTH] = m_Ranges[j][R_MAX] - m_Ranges[j][R_MIN];
                }
            }
        }
    }

    private void initNumericNominalAttributes(InstanceInformation info) {
        ArrayList<Integer> tmpNumericAttributes = new ArrayList<>();
        ArrayList<Integer> tmpNominalAttributes = new ArrayList<>();

        for (int i = 0; i < this.listAttributes.length; i++) {
            if (info.attribute(i).isNominal())
                tmpNominalAttributes.add(listAttributes[i]);
            else
                tmpNumericAttributes.add(listAttributes[i]);
        }
        nominalAttributes = new int[tmpNominalAttributes.size()];
        for (int i = 0; i < tmpNominalAttributes.size(); i++) {
            nominalAttributes[i] = tmpNominalAttributes.get(i);
        }
        numericAttributes = new int[tmpNumericAttributes.size()];
        for (int i = 0; i < tmpNumericAttributes.size(); i++) {
            numericAttributes[i] = tmpNumericAttributes.get(i);
        }
    }

    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        try {
            this.stm = new Instances(context, 0);
            this.stm.setClassIndex(context.classIndex());
            this.ltm = new Instances(context, 0);
            this.ltm.setClassIndex(context.classIndex());
            this.init();

            this.numAttributes = context.getInstanceInformation().numAttributes() - 1;
            this.listAttributes = new int[this.numAttributes];

            for (int j = 0; j < this.numAttributes; j++) {
                listAttributes[j] = j;
            }
            initNumericNominalAttributes(context.getInstanceInformation());

        } catch (Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void randomizeFeatures(int nAttributes, InstanceInformation info, Random rnd) {
        this.numAttributes = nAttributes;
        this.listAttributes = new int[this.numAttributes];
        for (int j = 0; j < this.numAttributes; j++) {
            boolean isUnique = false;
            while (!isUnique) {
                this.listAttributes[j] = rnd.nextInt(info.numAttributes() - 1);
                isUnique = true;
                for (int i = 0; i < j; i++) {
                    if (this.listAttributes[j] == this.listAttributes[i]) {
                        isUnique = false;
                        break;
                    }
                }
            }
        }
        initNumericNominalAttributes(info);
    }

    @Override
    public void resetLearningImpl() {
        this.stm = null;
        this.ltm = null;
        this.stmHistory = null;
        this.ltmHistory = null;
        this.cmHistory = null;
        this.distMSTM = null;

        this.predictionHistories = null;
        this.listAttributes = null;
    }

    private void shiftDistMIdx(int numShifts){
        distMStartIdx += numShifts;
    }

    private void checkForRewriteDistIdx(){
        int limitOptionValue = this.limitOption.getValue();
        if (distMStartIdx + stm.size()-1 >= limitOptionValue){
            for (int i = 0; i < this.stm.numInstances()-1; i++) {
                System.arraycopy(this.distMSTM[getDistancesSTMIdx(i)], distMStartIdx, this.distMSTM[i], 0, this.stm.numInstances()-1);
            }
            distMStartIdx = 0;
        }

    }

    private int getDistancesSTMIdx(int instanceIdx){
        return distMStartIdx + instanceIdx;
    }
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        long start = System.nanoTime();
        if (normalizeDistancesOption.isSet()) {
            if (m_Ranges == null)
                initRangesFirst(inst);
            else
                updateRanges(inst);
        }
        this.trainStepCount++;
        if (inst.classValue() > maxClassValue)
            maxClassValue = (int) inst.classValue();
        this.stm.addAsReference(inst);
        memorySizeCheck();
        clean(this.stm, this.ltm, true);

        checkForRewriteDistIdx();
        float distancesSTM[];
        int lastInstanceidx = getDistancesSTMIdx(this.stm.numInstances() - 1);

        if (inst == lastVotedInstance) {
            distancesSTM = lastVotedInstanceDistancesSTM;
            System.arraycopy(distancesSTM, 0, this.distMSTM[lastInstanceidx], getDistancesSTMIdx(0), this.stm.numInstances() - 1);
            this.distMSTM[lastInstanceidx][lastInstanceidx] = 0;
        } else {
            distancesSTM = this.get1ToNDistances(inst, this.stm);
            System.arraycopy(distancesSTM, 0, this.distMSTM[lastInstanceidx], getDistancesSTMIdx(0), this.stm.numInstances());
        }

        int oldWindowSize = this.stm.numInstances();
        int newWindowSize = this.getNewSTMSize(recalculateSTMErrorOption.isSet());

        if (newWindowSize < oldWindowSize) {
            int diff = oldWindowSize - newWindowSize;
            Instances discardedSTMInstances = new Instances(this.stm, 0);

            for (int i = diff; i>0;i--){
                discardedSTMInstances.addAsReference(this.stm.get(0));
                this.stm.delete(0);
            }
            shiftDistMIdx(diff);

            for (int i = 0; i < diff; i++) {
                this.stmHistory.remove(0);
                this.ltmHistory.remove(0);
                this.cmHistory.remove(0);
                if (stmMasterIndices != null)
                    this.stmMasterIndices.remove(0);
            }

            this.clean(this.stm, discardedSTMInstances, false);
            for (int i = 0; i < discardedSTMInstances.numInstances(); i++){
                this.ltm.addAsReference(discardedSTMInstances.get(i));
            }
            memorySizeCheck();
        }
        runTimeMeasurementTrain += System.nanoTime() - start;
    }


    /**
     * Predicts the label of a given sample by using the STM, LTM and the CM.
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {
        long start = System.nanoTime();
        double vSTM[];
        double vLTM[];
        double vCM[];
        double v[];
        float distancesSTM[];
        float distancesLTM[];
        int predClassSTM = 0;
        int predClassLTM = 0;
        int predClassCM = 0;
        try {
            if (this.stm.numInstances() > 0) {
                distancesSTM = get1ToNDistances2(inst, this.stm);
                lastVotedInstance = inst;
                lastVotedInstanceDistancesSTM = distancesSTM;
                int nnIndicesSTM[] = nArgMin(Math.min(distancesSTM.length, this.kOption.getValue()), distancesSTM);
                vSTM = getDistanceWeightedVotes(distancesSTM, nnIndicesSTM, this.stm, 0);
                predClassSTM = this.getClassFromVotes(vSTM);
                distancesLTM = get1ToNDistances(inst, this.ltm);
                lastVotedInstanceDistancesLTM = distancesLTM;


                if (this.ltm.numInstances() >= 0) {
                    int nnIndicesLTM[] = nArgMin(Math.min(distancesLTM.length, this.kOption.getValue()), distancesLTM);
                    vLTM = getDistanceWeightedVotes(distancesLTM, nnIndicesLTM, this.ltm, 0);
                    predClassLTM = this.getClassFromVotes(vLTM);
                } else {
                    vLTM = new double[inst.numClasses()];
                }
                vCM = getCMVotes(distancesSTM, this.stm, distancesLTM, this.ltm);
                predClassCM = this.getClassFromVotes(vCM);

                int correctSTM = historySum(this.stmHistory);
                int correctLTM = historySum(this.ltmHistory);
                int correctCM = historySum(this.cmHistory);
                if (correctSTM >= correctLTM && correctSTM >= correctCM) {
                    v = vSTM;
                    this.accCurrentConcept = correctSTM / (float) this.stmHistory.size();
                } else if (correctLTM > correctSTM && correctLTM >= correctCM) {
                    v = vLTM;
                    this.accCurrentConcept = correctLTM / (float) this.stmHistory.size();
                } else {
                    v = vCM;
                    this.accCurrentConcept = correctCM / (float) this.stmHistory.size();
                }
            } else {
                v = new double[inst.numClasses()];
                this.accCurrentConcept = 1 / (float) inst.numClasses();
            }
            this.stmHistory.add((predClassSTM == inst.classValue()) ? 1 : 0);
            this.ltmHistory.add((predClassLTM == inst.classValue()) ? 1 : 0);
            this.cmHistory.add((predClassCM == inst.classValue()) ? 1 : 0);

        } catch (Exception e) {
            return new double[inst.numClasses()];
        }
        runTimeMeasurementVotes += System.nanoTime() - start;
        return v;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return true;
    }


    private int historySum(List<Integer> history) {
        int sum = 0;
        for (Integer e : history) {
            sum += e;
        }
        return sum;
    }

    private List<double[]> kMeans(List<double[]> points, int k) {
        List<double[]> centroids = CoresetKMeans
                .generatekMeansPlusPlusCentroids(k,
                        points, this.classifierRandom);
        CoresetKMeans.kMeans(centroids, points);
        return centroids;
    }

    /**
     * Performs classwise kMeans++ clustering for given samples with corresponding labels. The number of samples is halved per class.
     */
    private void clusterDown() {
        int classIndex = this.ltm.classIndex();
        for (int c = 0; c <= this.maxClassValue; c++) {
            List<double[]> classSamples = new ArrayList<>();
            for (int i = this.ltm.numInstances() - 1; i > -1; i--) {
                if (this.ltm.get(i).classValue() == c) {
                    classSamples.add(this.ltm.get(i).toDoubleArray());
                    this.ltm.delete(i);
                }
            }
            if (classSamples.size() > 0) {
                //used kMeans++ implementation expects the weight of each sample at the first index,
                // make sure that the first value gets the uniform weight 1, overwrite class value
                for (double[] sample : classSamples) {
                    if (classIndex != 0) {
                        sample[classIndex] = sample[0];
                    }
                    sample[0] = 1;
                }

                List<double[]> centroids = this.kMeans(classSamples, Math.max(classSamples.size() / 2, 1));

                for (double[] centroid : centroids) {

                    double[] attributes = new double[this.ltm.numAttributes()];
                    //returned centroids do not contain the weight anymore, but simply the data
                    System.arraycopy(centroid, 0, attributes, 1, this.ltm.numAttributes() - 1);
                    //switch back if necessary
                    if (classIndex != 0) {
                        attributes[0] = attributes[classIndex];
                    }
                    attributes[classIndex] = c;
                    Instance inst = new InstanceImpl(1, attributes);
                    inst.setDataset(this.ltm);
                    this.ltm.addAsReference(inst);
                }
            }

        }
        lastVotedInstanceDistancesLTM = null;
    }

    /**
     * Makes sure that the STM and LTM combined doe not surpass the maximum size.
     */

    private void memorySizeCheck() {
        if (this.stm.numInstances() + this.ltm.numInstances() > this.maxSTMSize + this.maxLTMSize) {
            if (this.ltm.numInstances() > this.maxLTMSize) {
                this.clusterDown();
            } else { //shift values from STM directly to LTM since STM is full
                int numShifts = this.maxLTMSize - this.ltm.numInstances() + 1;
                for (int i = 0; i < numShifts; i++) {
                    this.ltm.addAsReference(this.stm.get(0));
                    this.stm.delete(0);
                    this.stmHistory.remove(0);
                    this.ltmHistory.remove(0);
                    this.cmHistory.remove(0);
                    if (stmMasterIndices != null)
                        this.stmMasterIndices.remove(0);
                }
                this.clusterDown();
                this.predictionHistories.clear();
                shiftDistMIdx(numShifts);
            }
        }
    }

    private void cleanSingle(Instances cleanAgainst, int cleanAgainstindex, Instances toClean, float[] _distancesLTM) {
        Instances cleanAgainstTmp = new Instances(cleanAgainst);
        cleanAgainstTmp.delete(cleanAgainstindex);

        float distancesSTM[] = new float[cleanAgainstTmp.size()];
        float _distSTM[] = distMSTM[getDistancesSTMIdx(cleanAgainstindex)];

        if (cleanAgainstindex != 0)
            System.arraycopy(_distSTM, getDistancesSTMIdx(0), distancesSTM, 0, cleanAgainstindex);
        if (cleanAgainstindex != cleanAgainstTmp.size())
            System.arraycopy(_distSTM, getDistancesSTMIdx(cleanAgainstindex + 1), distancesSTM, cleanAgainstindex, cleanAgainstTmp.size() - cleanAgainstindex);

        int nnIndicesSTM[] = nArgMin(Math.min(this.kOption.getValue(), distancesSTM.length), distancesSTM);

        float distancesLTM[];
        if (lastVotedInstance == cleanAgainst.get(cleanAgainstindex) && _distancesLTM != null){
            distancesLTM = _distancesLTM;
        }
        else{
            distancesLTM = get1ToNDistances(cleanAgainst.get(cleanAgainstindex), toClean);
        }


        int nnIndicesLTM[] = nArgMin(Math.min(this.kOption.getValue(), distancesLTM.length), distancesLTM);
        double distThreshold = 0;
        for (int nnIdx : nnIndicesSTM) {
            if (cleanAgainstTmp.get(nnIdx).classValue() == cleanAgainst.get(cleanAgainstindex).classValue()) {
                if (distancesSTM[nnIdx] > distThreshold) {
                    distThreshold = distancesSTM[nnIdx];
                }
            }
        }

        List<Integer> delIndices = new ArrayList<>();
        for (int nnIdx : nnIndicesLTM) {
            if (toClean.get(nnIdx).classValue() != cleanAgainst.get(cleanAgainstindex).classValue()) {
                if (distancesLTM[nnIdx] <= distThreshold) {
                    delIndices.add(nnIdx);
                }
            }
        }
        Collections.sort(delIndices, Collections.reverseOrder());
        for (Integer idx : delIndices) {
            toClean.delete(idx);
        }
    }

    /**
     * Removes distance-based all instances from the input samples that contradict those in the STM.
     */
    private void clean(Instances cleanAgainst, Instances toClean, boolean onlyLast) {
        //long start = System.nanoTime();
        if (cleanAgainst.numInstances() > this.kOption.getValue() && toClean.numInstances() > 0) {
            if (onlyLast) {
                cleanSingle(cleanAgainst, (cleanAgainst.numInstances() - 1), toClean, lastVotedInstanceDistancesLTM);
            } else {
                for (int i = 0; i < cleanAgainst.numInstances(); i++) {
                    cleanSingle(cleanAgainst, i, toClean, null);
                }
            }
        }
        //runTimeMeasurement += System.nanoTime() - start;
    }

    /**
     * Returns the distance weighted votes.
     */
    private double[] getDistanceWeightedVotes(float distances[], int[] nnIndices, Instances instances, int startIdx) {

        double v[] = new double[this.maxClassValue + 1];
        if (this.uniformWeightedOption.isSet()) {
            for (int nnIdx : nnIndices) {
                v[(int) instances.instance(nnIdx).classValue()] += 1;
            }
        } else {
            for (int nnIdx : nnIndices) {
                v[(int) instances.instance(nnIdx-startIdx).classValue()] += 1. / Math.max(distances[nnIdx], 0.000000001);
            }
        }
        return v;
    }


    private double[] getDistanceWeightedVotesCM(float distances[], int[] nnIndices, Instances stm, Instances ltm) {
        double v[] = new double[this.maxClassValue + 1];
        if (this.uniformWeightedOption.isSet()) {
            for (int nnIdx : nnIndices) {
                if (nnIdx < stm.numInstances()) {
                    v[(int) stm.instance(nnIdx).classValue()] += 1;
                } else {
                    v[(int) ltm.instance((nnIdx - stm.numInstances())).classValue()] += 1;
                }
            }
        } else {
            for (int nnIdx : nnIndices) {
                if (nnIdx < stm.numInstances()) {
                    v[(int) stm.instance(nnIdx).classValue()] += 1. / Math.max(distances[nnIdx], 0.000000001);
                } else {
                    v[(int) ltm.instance((nnIdx - stm.numInstances())).classValue()] += 1. / Math.max(distances[nnIdx], 0.000000001);
                }
            }
        }
        return v;
    }

    /**
     * Returns the distance weighted votes for the combined memory (CM).
     */
    private double[] getCMVotes(float distancesSTM[], Instances stm, float distancesLTM[], Instances ltm) {
        //long start = System.nanoTime();
        float[] distancesCM = new float[distancesSTM.length + distancesLTM.length];
        System.arraycopy(distancesSTM, 0, distancesCM, 0, distancesSTM.length);
        System.arraycopy(distancesLTM, 0, distancesCM, distancesSTM.length, distancesLTM.length);
        int nnIndicesCM[] = nArgMin(Math.min(distancesCM.length, this.kOption.getValue()), distancesCM);
        double[] votes = getDistanceWeightedVotesCM(distancesCM, nnIndicesCM, stm, ltm);
        //runTimeMeasurement += System.nanoTime() - start;
        return votes;

    }

    /**
     * Returns the class with maximum vote.
     */
    private int getClassFromVotes(double votes[]) {
        double maxVote = -1;
        int maxVoteClass = -1;
        for (int i = 0; i < votes.length; i++) {
            if (votes[i] > maxVote) {
                maxVote = votes[i];
                maxVoteClass = i;
            }
        }
        return maxVoteClass;
    }

    private int getLabelFct(float distances[], Instances instances, int startIdx, int endIdx) {
        int nnIndices[] = nArgMin(Math.min(this.kOption.getValue(), instances.size()), distances, startIdx, endIdx);
        double votes[] = getDistanceWeightedVotes(distances, nnIndices, instances, this.distMStartIdx);
        return this.getClassFromVotes(votes);
    }

    protected double norm(double x, int i) {
        if (m_Ranges[i][R_MAX] == m_Ranges[i][R_MIN]) {
            return 0;
        }
        else
            return (x - m_Ranges[i][R_MIN]) / (m_Ranges[i][R_WIDTH]);
    }

    private float getEuclideanDistance(Instance sample, Instance sample2) {
        float sum = 0;

        for (int i = 0; i < nominalAttributes.length; i++) {
            if ((int) sample.valueInputAttribute(nominalAttributes[i]) != (int) sample2.valueInputAttribute(nominalAttributes[i])) {
                sum++;
            }
        }
        if (normalizeDistancesOption.isSet()) {
            for (int i = 0; i < numericAttributes.length; i++) {
                float diff = (float) (norm(sample.valueInputAttribute(numericAttributes[i]), numericAttributes[i]) - norm(sample2.valueInputAttribute(numericAttributes[i]), numericAttributes[i]));
                sum += diff * diff;
            }
        }
        else {
            for (int i = 0; i < numericAttributes.length; i++) {
                float diff = (float) (sample.valueInputAttribute(numericAttributes[i]) - sample2.valueInputAttribute(numericAttributes[i]));
                sum += diff * diff;
            }
        }
        return (float) Math.sqrt(sum);
    }

    private float getManhattenDistance(Instance sample, Instance sample2, int[] listAttributes) {

        float sum = 0;

        for (int i = 0; i < listAttributes.length; i++) {
            float diff = (float) Math.abs((sample.valueInputAttribute(listAttributes[i]) - sample2.valueInputAttribute(listAttributes[i])));
            sum += diff;
        }
        return sum;
    }

    private float getChebychevDistance(Instance sample, Instance sample2, int[] listAttributes) {
        float maxDiff = Float.MAX_VALUE;

        for (int i = 0; i < listAttributes.length; i++) {
            float diff = (float) Math.abs((sample.valueInputAttribute(listAttributes[i]) - sample2.valueInputAttribute(listAttributes[i])));
            if (diff > maxDiff)
                maxDiff = diff;
        }
        return maxDiff;
    }

    private float getDistance(Instance sample, Instance sample2) {
        if (distanceMetricOption.getChosenLabel() == rManhatten) {
            return getManhattenDistance(sample, sample2, listAttributes);
        } else if (distanceMetricOption.getChosenLabel() == rChebychev) {
            return getChebychevDistance(sample, sample2, listAttributes);
        } else {
            return getEuclideanDistance(sample, sample2);
        }
    }

    private float[] get1ToNDistances2(Instance sample, Instances samples) {
        float[] distances = null;
        if (STMDistMatrix != null) {
            //System.out.println("stuff");
            distances = STMDistMatrix.getSTMDistancesToIndices(sample, stmMasterIndices);
            //System.out.println(distances.length);
        }
        if (distances == null)
            distances = get1ToNDistances(sample, samples);
        return distances;
    }


    /**
     * Returns the Euclidean distance between one sample and a collection of samples in an 1D-array.
     */
    public float[] get1ToNDistances(Instance sample, Instances samples) {
        long start = System.nanoTime();
        float distances[] = new float[samples.numInstances()];
        for (int i = 0; i < samples.numInstances(); i++) {
            distances[i] = this.getDistance(sample, samples.get(i));
        }
        runTimeMeasurement += System.nanoTime() - start;
        return distances;
    }

    /**
     * Returns the n smallest indices of the smallest values (sorted).
     */
    private int[] nArgMin(int n, float[] values, int startIdx, int endIdx) {
        int indices[] = new int[n];
        for (int i = 0; i < n; i++) {
            float minValue = Float.MAX_VALUE;
            for (int j = startIdx; j < endIdx + 1; j++) {
                if (values[j] < minValue) {
                    boolean alreadyUsed = false;
                    for (int k = 0; k < i; k++) {
                        if (indices[k] == j) {
                            alreadyUsed = true;
                            break;
                        }
                    }
                    if (!alreadyUsed) {
                        indices[i] = j;
                        minValue = values[j];
                    }
                }
            }
        }
        return indices;
    }

    private int[] nArgMin(int n, float[] values) {
        return nArgMin(n, values, 0, values.length - 1);
    }

    /**
     * Removes predictions of the largest window size and shifts the remaining ones accordingly.
     */
    private void adaptHistories(int numberOfDeletions) {
        for (int i = 0; i < numberOfDeletions; i++) {
            SortedSet<Integer> keys = new TreeSet<>(this.predictionHistories.keySet());
            this.predictionHistories.remove(keys.first());
            keys = new TreeSet<>(this.predictionHistories.keySet());
            for (Integer key : keys) {
                List<Integer> predHistory = this.predictionHistories.remove(key);
                this.predictionHistories.put(key - keys.first(), predHistory);
            }
        }
    }

    /**
     * Creates a prediction history incrementally by using the previous predictions.
     */
    private List<Integer> getIncrementalTestTrainPredHistory(Instances instances, int startIdx, List<Integer> predictionHistory) {
        for (int i = startIdx + this.kOption.getValue() + predictionHistory.size(); i < instances.numInstances(); i++) {
            predictionHistory.add((this.getLabelFct(distMSTM[getDistancesSTMIdx(i)], instances, getDistancesSTMIdx(startIdx), getDistancesSTMIdx(i - 1)) == instances.get(i).classValue()) ? 1 : 0);
        }
        return predictionHistory;
    }

    /**
     * Creates a prediction history from the scratch.
     */
    private List<Integer> getTestTrainPredHistory(Instances instances, int startIdx) {
        List<Integer> predictionHistory = new ArrayList<>();
        for (int i = startIdx + this.kOption.getValue(); i < instances.numInstances(); i++) {
            predictionHistory.add((this.getLabelFct(distMSTM[getDistancesSTMIdx(i)], instances, getDistancesSTMIdx(startIdx), getDistancesSTMIdx(i - 1)) == instances.get(i).classValue()) ? 1 : 0);
        }
        return predictionHistory;
    }

    /**
     * Returns the window size with the minimum Interleaved test-train error, using bisection (with recalculation of the STM error).
     */
    private int getMinErrorRateWindowSize() {

        int numSamples = this.stm.numInstances();
        if (numSamples < 2 * this.minSTMSizeOption.getValue()) {
            return numSamples;
        } else {
            List<Integer> numSamplesRange = new ArrayList<>();
            numSamplesRange.add(numSamples);
            while (numSamplesRange.get(numSamplesRange.size() - 1) >= 2 * this.minSTMSizeOption.getValue())
                numSamplesRange.add(numSamplesRange.get(numSamplesRange.size() - 1) / 2);

            Iterator it = this.predictionHistories.keySet().iterator();
            while (it.hasNext()) {
                Integer key = (Integer) it.next();
                if (!numSamplesRange.contains(numSamples - key)) {
                    it.remove();
                }
            }
            List<Double> errorRates = new ArrayList<>();
            for (Integer numSamplesIt : numSamplesRange) {
                int idx = numSamples - numSamplesIt;
                List<Integer> predHistory;
                if (this.predictionHistories.containsKey(idx)) {
                    predHistory = this.getIncrementalTestTrainPredHistory(this.stm, idx, this.predictionHistories.get(idx));
                } else {
                    predHistory = this.getTestTrainPredHistory(this.stm, idx);
                }
                this.predictionHistories.put(idx, predHistory);
                errorRates.add(this.getHistoryErrorRate(predHistory));
            }
            int minErrorRateIdx = errorRates.indexOf(Collections.min(errorRates));
            int windowSize = numSamplesRange.get(minErrorRateIdx);
            if (windowSize < numSamples) {
                this.adaptHistories(minErrorRateIdx);
            }
            return windowSize;
        }
    }

    /**
     * Calculates the achieved error rate of a history.
     */
    private double getHistoryErrorRate(List<Integer> predHistory) {
        double sumCorrect = 0;
        for (Integer e : predHistory) {
            sumCorrect += e;
        }
        return 1. - (sumCorrect / predHistory.size());
    }

    /**
     * Returns the window size with the minimum Interleaved test-train error, using bisection (without recalculation using an incremental approximation).
     */
    private int getMinErrorRateWindowSizeIncremental() {
        int numSamples = this.stm.numInstances();
        if (numSamples < 2 * this.minSTMSizeOption.getValue()) {
            return numSamples;
        } else {
            List<Integer> numSamplesRange = new ArrayList<>();
            numSamplesRange.add(numSamples);
            while (numSamplesRange.get(numSamplesRange.size() - 1) >= 2 * this.minSTMSizeOption.getValue())
                numSamplesRange.add(numSamplesRange.get(numSamplesRange.size() - 1) / 2);
            List<Double> errorRates = new ArrayList<>();
            for (Integer numSamplesIt : numSamplesRange) {
                int idx = numSamples - numSamplesIt;
                List<Integer> predHistory;
                if (this.predictionHistories.containsKey(idx)) {
                    predHistory = this.getIncrementalTestTrainPredHistory(this.stm, idx, this.predictionHistories.get(idx));
                } else if (this.predictionHistories.containsKey(idx - 1)) {
                    predHistory = this.predictionHistories.remove(idx - 1);
                    predHistory.remove(0);
                    predHistory = this.getIncrementalTestTrainPredHistory(this.stm, idx, predHistory);
                    this.predictionHistories.put(idx, predHistory);
                } else {
                    predHistory = this.getTestTrainPredHistory(this.stm, idx);
                    this.predictionHistories.put(idx, predHistory);
                }
                errorRates.add(this.getHistoryErrorRate(predHistory));
            }

            int minErrorRateIdx = errorRates.indexOf(Collections.min(errorRates));
            if (minErrorRateIdx > 0) {
                for (int i = 1; i < errorRates.size(); i++) {
                    if (errorRates.get(i) < errorRates.get(0)) {
                        int idx = numSamples - numSamplesRange.get(i);
                        List<Integer> predHistory = this.getTestTrainPredHistory(this.stm, idx);
                        errorRates.set(i, this.getHistoryErrorRate(predHistory));
                        this.predictionHistories.remove(idx);
                        this.predictionHistories.put(idx, predHistory);
                    }
                }
                minErrorRateIdx = errorRates.indexOf(Collections.min(errorRates));
            }
            int windowSize = numSamplesRange.get(minErrorRateIdx);
            if (windowSize < numSamples) {
                this.adaptHistories(minErrorRateIdx);
            }
            return windowSize;
        }
    }

    /**
     * Returns the bisected STM size which minimizes the interleaved-test-train error.
     */
    private int getNewSTMSize(boolean recalculateErrors) {
        if (recalculateErrors)
            return this.getMinErrorRateWindowSize();
        else
            return this.getMinErrorRateWindowSizeIncremental();
    }

    public interface STMDistanceMatrix{
        float[] getSTMDistancesToIndices(Instance sample, List<Integer> indices);

    }

}
