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
import moa.classifiers.lazy.SAMkNN;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;

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

    public StringOption uuidOption = new StringOption("uuidPrefix", 'o',
            "uuidPrefix.", "");
    protected Classifier[] ensemble;
    private ArrayList<Integer>[] ensembleLabels;
    private double lamdas[];

    protected int numberOfChangesDetected;

    protected int[][] matrixCodes;

    protected boolean initMatrixCodes = false;
    int noCount = 0;
    protected ADWIN[] ADError;
    protected ADWIN adwin;

    public IntOption eraseMembers = new IntOption("eraseMembers", 'y',
            "eraseMembers", 0, 0, 2);

    public FlagOption useAdwin = new FlagOption("useAdwin", 'n',
            "useAdwin");

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


    public void randomizeEnsembleMember(SAMkNNFS member, int index) {
        member.randomMeta = this.classifierRandom;
        if (randomizeK.isSet()){
            member.kOption.setValue(this.classifierRandom.nextInt(7)+1);
            System.out.println(member.kOption.getValue());
        }
        if (randomizeFeatures.isSet()){
            member.randomizeFeatures = true;
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
    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i].setModelContext(context);
            if (this.ensemble[i] instanceof SAMkNNFS){
                randomizeEnsembleMember((SAMkNNFS)this.ensemble[i], i);
            }
        }
    }

    @Override
    public void resetLearningImpl() {
        System.out.println("resetLearningImpl");
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        ensembleLabels = new ArrayList[this.ensembleSizeOption.getValue()];
        lamdas = new double[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);

        this.adwin = new ADWIN(this.deltaAdwinOption.getValue());

        if (useAdwin.isSet()) {
            this.ADError = new ADWIN[this.ensemble.length];
            for (int i = 0; i < this.ensemble.length; i++) {
                this.ADError[i] = new ADWIN((double) this.deltaAdwinOption.getValue());
            }
        }
        for (int i = 0; i < this.ensemble.length; i++) {
            lamdas[i] = weightShrinkOption.getValue();
            ensembleLabels[i] = new ArrayList();
            this.ensemble[i] = baseLearner.copy();
            //this.ensemble[i].setRandomSeed(i*100);
            this.ensemble[i].resetLearning();
            //System.out.println(((SAMkNN2)this.ensemble[i]).limitOption.getValue() + " " + ((SAMkNN2)this.ensemble[i]).minSTMSizeOption.getValue() + " " +                    ((SAMkNN2)this.ensemble[i]).randomSeed);
        }
        this.numberOfChangesDetected = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        boolean Change = false;
        trainstepCount ++;
        Instance weightedInst = (Instance) inst.copy();
        boolean correct = this.correctlyClassifies(inst);

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
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }else {
                noCount++;
                //System.out.println(noCount);
            }
            if (useAdwin.isSet()) {
                boolean correctlyClassifies = this.ensemble[i].correctlyClassifies(weightedInst);
                double ErrEstim = this.ADError[i].getEstimation();
                if (this.ADError[i].setInput(correctlyClassifies ? 0 : 1)) {
                    if (this.ADError[i].getEstimation() > ErrEstim) {
                        Change = true;
                    }
                }
            }
        }
        if (Change) {
            numberOfChangesDetected++;
            double max = 0.0;
            int imax = -1;
            for (int i = 0; i < this.ensemble.length; i++) {
                if (max < this.ADError[i].getEstimation()) {
                    max = this.ADError[i].getEstimation();
                    imax = i;
                }
            }
            if (imax != -1) {
                System.out.println("change and remove " + imax);
                this.ensemble[imax].resetLearning();
                this.ensemble[imax].setModelContext(this.modelContext);
                if (this.ensemble[imax] instanceof SAMkNNFS){
                    randomizeEnsembleMember((SAMkNNFS)this.ensemble[imax], imax);
                }

                //this.ensemble[imax].trainOnInstance(inst);
                this.ADError[imax] = new ADWIN((double) this.deltaAdwinOption.getValue());
            }
        }
        double ErrEstim = 0;
        if (eraseMembers.getValue() == 2)
            ErrEstim = this.adwin.getEstimation();

        if ((eraseMembers.getValue() == 1 && trainstepCount%2000 == 0) || (eraseMembers.getValue() == 2 && this.adwin.setInput(correct ? 0 : 1) && this.adwin.getEstimation() > ErrEstim)){
            if (this.ensemble[0] instanceof SAMkNNFS){
                double max = 0.0;
                int imax = -1;
                for (int i = 0; i < this.ensemble.length; i++) {
                    double error = 1 - ((SAMkNNFS) this.ensemble[i]).accCurrentConcept;
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
                    if (this.ensemble[imax] instanceof SAMkNNFS){
                        randomizeEnsembleMember((SAMkNNFS)this.ensemble[imax], imax);
                    }
                }
            }
        }

    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            double[] voteTmp = this.ensemble[i].getVotesForInstance(inst);
            DoubleVector vote = new DoubleVector(voteTmp);
            ensembleLabels[i].add(moa.core.Utils.maxIndex(voteTmp));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                if (this.ensemble[i] instanceof SAMkNNFS) {
                    double acc = ((SAMkNNFS) this.ensemble[i]).accCurrentConcept;
                    if (!this.disableWeightedVote.isSet() && acc > 0.0) {
                        for (int v = 0; v < vote.numValues(); ++v) {
                            vote.setValue(v, vote.getValue(v) * acc);
                        }
                    }
                    combinedVote.addValues(vote);
                }else{
                    for (int v = 0; v < vote.numValues(); ++v) {
                        vote.setValue(v, vote.getValue(v));
                    }
                    combinedVote.addValues(vote);
                }
            }
        }
        return combinedVote.getArrayRef();
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
}

