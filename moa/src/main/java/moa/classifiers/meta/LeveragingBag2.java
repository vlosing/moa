/*
 *    LeveragingBag.java
 *    Copyright (C) 2010 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceInformation;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.lazy.SAMkNNFS;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.List;
import java.util.concurrent.*;

/**
 * Leveraging Bagging for evolving data streams using ADWIN. Leveraging Bagging
 * and Leveraging Bagging MC using Random Output Codes ( -o option).
 *
 * <p>See details in:<br /> Albert Bifet, Geoffrey Holmes, Bernhard Pfahringer.
 * Leveraging Bagging for Evolving Data Streams Machine Learning and Knowledge
 * Discovery in Databases, European Conference, ECML PKDD}, 2010.</p>
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class LeveragingBag2 extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Leveraging Bagging for evolving data streams using ADWIN.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "lazy.SAMkNNFS");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public FloatOption weightShrinkOption = new FloatOption("weightShrink", 'w',
            "The number to use to compute the weight of new instances.", 6, 0.0, Float.MAX_VALUE);

    public FloatOption deltaAdwinOption = new FloatOption("deltaAdwin", 'a',
            "Delta of Adwin change detection", 0.002, 0.0, 1.0);


    public MultiChoiceOption leveraginBagAlgorithmOption = new MultiChoiceOption(
            "leveraginBagAlgorithm", 'm', "Leveraging Bagging to use.", new String[]{
                "LeveragingBag", "LeveragingBagME", "LeveragingBagHalf", "LeveragingBagWT", "LeveragingSubag"},
            new String[]{"Leveraging Bagging for evolving data streams using ADWIN",
                "Leveraging Bagging ME using weight 1 if misclassified, otherwise error/(1-error)",
                "Leveraging Bagging Half using resampling without replacement half of the instances",
                "Leveraging Bagging WT without taking out all instances.",
                "Leveraging Subagging using resampling without replacement."
            }, 0);
    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
            "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);

    public FlagOption asynchronousMode = new FlagOption("asynchronousMode", 'A',
            "Asynchronous multi-threading execution");

    public StringOption uuidOption = new StringOption("uuidPrefix", 'o',
            "uuidPrefix.", "");
    protected SAMkNNFS[] ensemble;
    private ArrayList<Integer>[] ensembleLabels;
    private double lamdas[];

    protected int numberOfChangesDetected;

    protected int[][] matrixCodes;

    protected boolean initMatrixCodes = false;
    int noCount = 0;

    protected ADWIN adwin;

    public IntOption eraseMembers = new IntOption("eraseMembers", 'y',
            "eraseMembers", 0, 0, 2);

    public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'd',
            "Should use weighted voting?");

    public FlagOption randomizeLamda = new FlagOption("randomizeLamda", 'z',
            "randomizeLamda");

    public FlagOption randomizeK = new FlagOption("randomizeK", 'k',
            "randomizeFeatures");

    public FlagOption randomizeFeatures = new FlagOption("randomizeFeatures", 'f',
            "randomizeFeatures");

    public FlagOption randomizeDistanceMetric = new FlagOption("randomizeDistanceMetric", 'e',
            "randomizeDistanceMetric");
    public FlagOption randomizWeighting = new FlagOption("randomizWeighting", 'g',
            "randomizeWeighting");

    private ExecutorService executor;


    public void randomizeEnsembleMember(SAMkNNFS member, int index, int numAttributes) {
        if (randomizeK.isSet()){
            member.kOption.setValue(this.classifierRandom.nextInt(7)+1);
            System.out.println(member.kOption.getValue());
        }
        if (randomizeFeatures.isSet()){
            int n = numAttributes-1;
            int nFeatures = Math.min((int) ((Math.round(n * 0.7) + 1)), n) ;
            member.randomizeFeatures(nFeatures, n, this.classifierRandom);
        }
        if (randomizeDistanceMetric.isSet()){
            member.distanceMetricOption.setChosenIndex(this.classifierRandom.nextInt(2));
            System.out.println(member.distanceMetricOption.getChosenLabel());
        }
        if (randomizWeighting.isSet()){
            if (this.classifierRandom.nextInt(2) == 1)
                member.uniformWeightedOption.set();
            System.out.println(member.uniformWeightedOption.isSet());
        }
        if (randomizeLamda.isSet()){
            lamdas[index] = Math.max(this.classifierRandom.nextDouble()*6, + 0.2);
            System.out.println(lamdas[index]);
        }
    }
    public int trainstepCount = 0;
    private Instance lastVotedInstance;
    private double[] lastVotes;
    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i].setModelContext(context);
            System.out.println(context.getInstanceInformation().numAttributes());
            randomizeEnsembleMember(this.ensemble[i], i, context.getInstanceInformation().numAttributes());
        }
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble = new SAMkNNFS[this.ensembleSizeOption.getValue()];
        ensembleLabels = new ArrayList[this.ensembleSizeOption.getValue()];
        lamdas = new double[this.ensembleSizeOption.getValue()];
        SAMkNNFS baseLearner = (SAMkNNFS) getPreparedClassOption(this.baseLearnerOption);

        this.adwin = new ADWIN(this.deltaAdwinOption.getValue());

        for (int i = 0; i < this.ensemble.length; i++) {
            lamdas[i] = weightShrinkOption.getValue();
            ensembleLabels[i] = new ArrayList();
            this.ensemble[i] = (SAMkNNFS) baseLearner.copy();
            //this.ensemble[i].setRandomSeed(i*100);
            this.ensemble[i].resetLearning();
            //System.out.println(((SAMkNN2)this.ensemble[i]).limitOption.getValue() + " " + ((SAMkNN2)this.ensemble[i]).minSTMSizeOption.getValue() + " " +                    ((SAMkNN2)this.ensemble[i]).randomSeed);
        }
        this.numberOfChangesDetected = 0;
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1)
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent.
        // this.executor will be null and not used...
        if(numberOfJobs > 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
    }
    @Override
    public void afterLearning(){
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i].afterLearning();
        }
        if (executor!=null){
            executor.shutdown();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainstepCount ++;
        boolean correct = this.correctlyClassifies(inst);
        Collection<TrainingRunnable> tasks = new ArrayList<>();
        //Train ensemble of classifiers
        for (int i = 0; i < this.ensemble.length; i++) {
            double w = lamdas[i];
            double k = 0.0;
            switch (this.leveraginBagAlgorithmOption.getChosenIndex()) {
                case 0: //LeveragingBag
                    k = MiscUtils.poisson(w, this.classifierRandom);
                    break;
                case 2: //LeveragingBagHalf
                    w = 1.0;
                    k = this.classifierRandom.nextBoolean() ? 0.0 : w;
                    break;
                case 3: //LeveragingBagWT
                    w = 1.0;
                    k = 1.0 + MiscUtils.poisson(w, this.classifierRandom);
                    break;
                case 4: //LeveragingSubag
                    w = 1.0;
                    k = MiscUtils.poisson(1, this.classifierRandom);
                    k = (k > 0) ? w : 0;
                    break;
            }
            if (k > 0) {
                if(this.executor != null) {
                    if (asynchronousMode.isSet())
                        this.executor.submit(new TrainingRunnable(this.ensemble[i], inst));
                    else
                        tasks.add(new TrainingRunnable(this.ensemble[i], inst));
                }
                else { // SINGLE_THREAD is in-place...
                    this.ensemble[i].trainOnInstance(inst);
                }
            }else {
                noCount++;
                //System.out.println(noCount);
            }
        }
        if(this.executor != null) {
            try {
                this.executor.invokeAll(tasks);
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex.getMessage());
            }
        }
        double ErrEstim = 0;
        if (eraseMembers.getValue() == 2)
            ErrEstim = this.adwin.getEstimation();

        if ((eraseMembers.getValue() == 1 && trainstepCount%2000 == 0) || (eraseMembers.getValue() == 2 && this.adwin.setInput(correct ? 0 : 1) && this.adwin.getEstimation() > ErrEstim)){
            double max = 0.0;
            int imax = -1;
            for (int i = 0; i < this.ensemble.length; i++) {
                double error = 1 - this.ensemble[i].accCurrentConcept;
                if (max < error) {
                    max = error;
                    imax = i;
                }
            }
            if (imax != -1) {
                System.out.println(trainstepCount + " remove " + imax);
                if (eraseMembers.getValue() == 2)
                    System.out.println("adwin width " + adwin.getWidth());
                this.ensemble[imax].resetLearning();
                this.ensemble[imax].setModelContext(this.modelContext);
                randomizeEnsembleMember(this.ensemble[imax], imax, inst.numAttributes());
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (lastVotedInstance == inst){
            return lastVotes;
        }
        else{
            DoubleVector combinedVote = new DoubleVector();
            if (executor == null) {
                for (int i = 0; i < this.ensemble.length; ++i) {
                    double[] voteTmp = this.ensemble[i].getVotesForInstance(inst);
                    DoubleVector vote = new DoubleVector(voteTmp);
                    if (vote.sumOfValues() > 0.0) {
                        vote.normalize();
                        double acc = this.ensemble[i].accCurrentConcept;
                        if (!this.disableWeightedVote.isSet() && acc > 0.0) {
                            for (int v = 0; v < vote.numValues(); ++v) {
                                vote.setValue(v, vote.getValue(v) * acc);
                            }
                        }
                        combinedVote.addValues(vote);
                    }
                }
            } else {
                //finish training
                Collection<VotingRunnable> tasks = new ArrayList<>();
                for (int i = 0; i < this.ensemble.length; ++i) {
                    VotingRunnable task = new VotingRunnable(this.ensemble[i],
                            inst);
                    tasks.add(task);
                }
                try {
                    List<Future<double[]>> futures = this.executor.invokeAll(tasks);
                    int idx = 0;
                    for (Future<double[]> future : futures) {
                        DoubleVector vote = new DoubleVector(future.get());
                        if (vote.sumOfValues() > 0.0) {
                            vote.normalize();
                            double acc = this.ensemble[idx].accCurrentConcept;
                            if (!this.disableWeightedVote.isSet() && acc > 0.0) {
                                for (int v = 0; v < vote.numValues(); ++v) {
                                    vote.setValue(v, vote.getValue(v) * acc);
                                }
                            }
                            combinedVote.addValues(vote);
                        }
                        idx++;
                    }
                } catch (InterruptedException|ExecutionException ex) {
                    throw new RuntimeException("Could not call invokeAll() on threads.");
                }
            }
            lastVotedInstance = inst;
            lastVotes = combinedVote.getArrayRef();
            return lastVotes;
        }
    }

    @Override
    public int measureByteSize() {
        if (!uuidOption.getValue().equals("")) {
            Map<String, String> env = System.getenv();
            String dir = env.get("LOCAL_STORAGE_DIR") + "/Tmp/";
            try {
                String fileName = dir + "moaStatistics_" + uuidOption.getValue() + ".csv";
                PrintWriter writer = new PrintWriter(new FileOutputStream(fileName, false));
                for (int i= 0; i < ensembleLabels.length; i++){
                    writer.println(Utils.arrayToString(this.ensembleLabels[i].toArray()));
                }
                writer.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        return super.measureByteSize();
    }
    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                    this.ensemble != null ? this.ensemble.length : 0),
                    new Measurement("change detections", this.numberOfChangesDetected)
                };
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }
    /***
     * Inner class to assist with the multi-thread execution.
     */
    protected class TrainingRunnable implements Callable<Integer> {
        final private SAMkNNFS learner;
        final private Instance instance;

        public TrainingRunnable(SAMkNNFS learner, Instance instance) {
            this.learner = learner;
            this.instance = instance;
        }

        @Override
        public Integer call() throws Exception {
            learner.trainOnInstance(this.instance);
            return 0;
        }
    }

    protected class VotingRunnable implements Callable<double[]> {
        final private SAMkNNFS learner;
        final private Instance instance;
        double[] votes;
        public VotingRunnable(SAMkNNFS learner, Instance instance) {
            this.learner = learner;
            this.instance = instance;
        }

        @Override
        public double[] call() throws Exception {
            votes = learner.getVotesForInstance(this.instance);
            return votes;
        }
    }
}

