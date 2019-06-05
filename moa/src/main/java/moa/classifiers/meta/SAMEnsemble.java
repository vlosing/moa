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
import moa.classifiers.lazy.SAMkNN;
import moa.core.*;
import moa.options.ClassOption;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.*;
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
public class SAMEnsemble extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Leveraging Bagging for evolving data streams using ADWIN.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", SAMkNN.class, "SAMkNN");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public FloatOption lambdaOption = new FloatOption("Lambda", 'w',
            "The Lambda parameter for Bagging.", 6, 0.0, Float.MAX_VALUE);

    //public FloatOption deltaAdwinOption = new FloatOption("deltaAdwin", 'a', "Delta of Adwin change detection", 0.002, 0.0, 1.0);


    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
            "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);


    public StringOption uuidOption = new StringOption("uuidPrefix", 'u',
            "uuidPrefix.", "");


    protected SAMkNN[] ensemble;
    private ArrayList<Integer>[] ensembleLabels;
    private double lamdas[];

    protected int numberOfChangesDetected;

    protected int[][] matrixCodes;

    protected boolean initMatrixCodes = false;

    protected ADWIN adwin;


    public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'd',
            "Disables the weighting of the learners according to their current performance.");

    //public FlagOption randomizeLamda = new FlagOption("randomizeLamda", 'z', "randomizeLamda");

    public FlagOption randomizeK = new FlagOption("randomizeK", 'k', "randomizeFeatures");

    public FlagOption noDriftDetection = new FlagOption("noDriftDetection", 'r', "noDriftDetection");


    public FlagOption randomizeFeatures = new FlagOption("randomizeFeatures", 'f', "randomizeFeatures");

    //public FlagOption randomizeDistanceMetric = new FlagOption("randomizeDistanceMetric", 'e', "randomizeDistanceMetric");

    private ExecutorService executor;

    public void randomizeEnsembleMember(SAMkNN member, int index, InstanceInformation info) {
        if (randomizeK.isSet()){
            member.kOption.setValue(this.classifierRandom.nextInt(7)+1);
        }
        if (randomizeFeatures.isSet()){
            int n = info.numAttributes()-1;
            int nFeatures = Math.min((int) ((Math.round(n * 0.7) + 1)), n) ;
            //int nFeatures = (int) Math.round(Math.sqrt(n)) + 1;
            member.randomizeFeatures(nFeatures, info, this.classifierRandom);
        }
        /*if (randomizeDistanceMetric.isSet()){
            member.distanceMetricOption.setChosenIndex(this.classifierRandom.nextInt(2));
            //System.out.println(member.distanceMetricOption.getChosenLabel());
        }
        if (randomizeLamda.isSet()){
            lamdas[index] = Math.max(this.classifierRandom.nextDouble() * 6, + 0.2);
            //System.out.println(lamdas[index]);
        }*/
    }
    public int trainstepCount = 0;
    private Instance lastVotedInstance;
    private double[] lastVotes;

    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i].setModelContext(context);
            randomizeEnsembleMember(this.ensemble[i], i, context.getInstanceInformation());
        }
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble = new SAMkNN[this.ensembleSizeOption.getValue()];
        ensembleLabels = new ArrayList[this.ensembleSizeOption.getValue()];
        lamdas = new double[this.ensembleSizeOption.getValue()];
        SAMkNN baseLearner = (SAMkNN) getPreparedClassOption(this.baseLearnerOption);

        this.adwin = new ADWIN();

        for (int i = 0; i < this.ensemble.length; i++) {
            lamdas[i] = lambdaOption.getValue();
            ensembleLabels[i] = new ArrayList();
            this.ensemble[i] = (SAMkNN) baseLearner.copy();
            //this.ensemble[i].setRandomSeed(i*100);
            this.ensemble[i].resetLearning();
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
    public void trainOnInstances(List<Example<Instance>> examples){
        trainstepCount ++;
        Collection<TrainingRunnable2> tasks = new ArrayList<>();
        //Train ensemble of classifiers
        for (int i = 0; i < this.ensemble.length; i++) {
            List<Example<Instance>> trainExamples = new ArrayList<>();
            for (Example<Instance> example: examples) {
                double w = lamdas[i];
                double k = MiscUtils.poisson(w, this.classifierRandom);
                if (k > 0) {
                    trainExamples.add(example);
                }
            }
            if(this.executor == null) {
                for (Example<Instance> example: trainExamples) {
                    this.ensemble[i].trainOnInstance(example.getData());
                }
            } else {
                tasks.add(new TrainingRunnable2(this.ensemble[i], examples));
            }
        }
        if(this.executor != null) {
            try {
                this.executor.invokeAll(tasks);
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex.getMessage());
            }
        }
        //double adwinError = this.adwin.getEstimation();
        //if (this.adwin.setInput(correct ? 0 : 1) && this.adwin.getEstimation() > adwinError){
        for (Example<Instance> example: examples) {
            boolean correct = this.correctlyClassifies(example.getData());
            if (!noDriftDetection.isSet() && this.adwin.setInput(correct ? 0 : 1)) {
                int nRemovals = Math.max(ensemble.length / 10, 1);
                //System.out.println(trainstepCount + " " + nRemovals + " removals, adwin width " + adwin.getWidth());
                List<Integer> excludeIndices = new ArrayList<>();
                for (int k = 0; k < nRemovals; k++) {
                    double max = 0.0;
                    int imax = -1;
                    for (int i = 0; i < this.ensemble.length; i++) {
                        double error = 1 - this.ensemble[i].accCurrentConcept;
                        if (max < error && excludeIndices.indexOf(i) == -1) {
                            max = error;
                            imax = i;
                        }
                    }
                    if (imax != -1) {
                        excludeIndices.add(imax);
                        this.ensemble[imax].resetLearning();
                        this.ensemble[imax].setModelContext(this.modelContext);
                        randomizeEnsembleMember(this.ensemble[imax], imax, this.modelContext.getInstanceInformation());
                    }
                }
            }
        }

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainstepCount ++;
        Collection<TrainingRunnable> tasks = new ArrayList<>();
        //Train ensemble of classifiers
        for (int i = 0; i < this.ensemble.length; i++) {
            double w = lamdas[i];
            double k = 0.0;
            k = MiscUtils.poisson(w, this.classifierRandom);
            if (k > 0)
            {
                if(this.executor != null) {
                    tasks.add(new TrainingRunnable(this.ensemble[i], inst));
                }
                else { // SINGLE_THREAD is in-place...
                    this.ensemble[i].trainOnInstance(inst);
                }
            }
        }
        if(this.executor != null) {
            try {
                this.executor.invokeAll(tasks);
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex.getMessage());
            }
        }
        //double adwinError = this.adwin.getEstimation();
        //if (this.adwin.setInput(correct ? 0 : 1) && this.adwin.getEstimation() > adwinError){

        boolean correct = this.correctlyClassifies(inst);
        if (!noDriftDetection.isSet() && this.adwin.setInput(correct ? 0 : 1)){
            int nRemovals = Math.max(ensemble.length / 10, 1);
            //System.out.println(trainstepCount + " " + nRemovals + " removals, adwin width " + adwin.getWidth());
            List<Integer> excludeIndices = new ArrayList<>();
            for (int k = 0; k < nRemovals; k++) {
                double max = 0.0;
                int imax = -1;
                for (int i = 0; i < this.ensemble.length; i++) {
                    double error = 1 - this.ensemble[i].accCurrentConcept;
                    if (max < error && excludeIndices.indexOf(i) == -1)  {
                        max = error;
                        imax = i;
                    }
                }
                if (imax != -1) {
                    excludeIndices.add(imax);
                    this.ensemble[imax].resetLearning();
                    this.ensemble[imax].setModelContext(this.modelContext);
                    randomizeEnsembleMember(this.ensemble[imax], imax, this.modelContext.getInstanceInformation());
                }
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (lastVotedInstance == inst){
            return lastVotes;
        }
        else{
            lastVotedInstance = inst;

            DoubleVector combinedVote = new DoubleVector();
            if (executor == null) {
                for (int i = 0; i < this.ensemble.length; ++i) {
                    double[] voteTmp = this.ensemble[i].getVotesForInstance(inst);
                    ensembleLabels[i].add(Utils.maxIndex(voteTmp));

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
                        double[] voteTmp = future.get();
                        ensembleLabels[idx].add(Utils.maxIndex(voteTmp));
                        DoubleVector vote = new DoubleVector(voteTmp);
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

            lastVotes = combinedVote.getArrayRef();
            return lastVotes;
        }
    }

    @Override
    public List<double[]> getVotesForInstances(List<Example<Instance>> examples){
        List<double[]> votes = new ArrayList<>();
        DoubleVector combinedVote = new DoubleVector();
        if (executor == null) {
            for (Example<Instance> ex: examples ) {
                for (int i = 0; i < this.ensemble.length; ++i) {
                    double[] voteTmp = this.ensemble[i].getVotesForInstance(ex.getData());
                    ensembleLabels[i].add(Utils.maxIndex(voteTmp));

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
                votes.add(combinedVote.getArrayRef());
            }
        } else {
            Collection<VotingRunnable2> tasks = new ArrayList<>();
            for (int i = 0; i < this.ensemble.length; ++i) {
                VotingRunnable2 task = new VotingRunnable2(this.ensemble[i],
                        examples);
                tasks.add(task);
            }
            try {
                List<Future<List<double[]>>> futures = this.executor.invokeAll(tasks);
                for (int i=0; i < examples.size(); i++){
                    int idx = 0;
                    for (Future<List<double[]>> future : futures) {
                        double[] voteTmp = future.get().get(i);
                        ensembleLabels[idx].add(Utils.maxIndex(voteTmp));
                        DoubleVector vote = new DoubleVector(voteTmp);
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
                    votes.add(combinedVote.getArrayRef());
                }
            } catch (InterruptedException|ExecutionException ex) {
                throw new RuntimeException("Could not call invokeAll() on threads.");
            }
        }
        return votes;
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
                    writer.println(Arrays.toString(this.ensembleLabels[i].toArray()).replace("[", "").replace("]", ""));
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
        final private SAMkNN learner;
        final private Instance instance;

        public TrainingRunnable(SAMkNN learner, Instance instance) {
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
        final private SAMkNN learner;
        final private Instance instance;
        double[] votes;
        public VotingRunnable(SAMkNN learner, Instance instance) {
            this.learner = learner;
            this.instance = instance;
        }

        @Override
        public double[] call() throws Exception {
            votes = learner.getVotesForInstance(this.instance);
            return votes;
        }
    }

    protected class TrainingRunnable2 implements Callable<Integer> {
        final private SAMkNN learner;
        final private List<Example<Instance>> instances;

        public TrainingRunnable2(SAMkNN learner, List<Example<Instance>>  instances) {
            this.learner = learner;
            this.instances = instances;
        }

        @Override
        public Integer call() throws Exception {
            for(Example<Instance> inst: this.instances) {
                learner.trainOnInstance(inst.getData());
            }
            return 0;
        }
    }

    protected class VotingRunnable2 implements Callable<List<double[]>> {
        final private SAMkNN learner;
        final private List<Example<Instance>> instances;
        public VotingRunnable2(SAMkNN learner, List<Example<Instance>>  instances) {
            this.learner = learner;
            this.instances = instances;
        }

        @Override
        public List<double[]> call() throws Exception {
            List<double[]> votes = new ArrayList<>();
            for(Example<Instance> inst: this.instances){
                votes.add(learner.getVotesForInstance(inst.getData()));
            }
            return votes;
        }
    }
}

