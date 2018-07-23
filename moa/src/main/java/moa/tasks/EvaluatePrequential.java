/*
 *    EvaluatePrequential.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
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
package moa.tasks;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import com.github.javacliparser.StringOption;
import moa.classifiers.Classifier;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.evaluation.EWMAClassificationPerformanceEvaluator;
import moa.evaluation.FadingFactorClassificationPerformanceEvaluator;
import moa.evaluation.LearningCurve;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.learners.Learner;
import moa.options.ClassOption;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.streams.ExampleStream;
import moa.streams.InstanceStream;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Utils;

/**
 * Task for evaluating a classifier on a stream by testing then training with each example in sequence.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class EvaluatePrequential extends MainTask {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by testing then training with each example in sequence.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Learner to train.", Classifier.class, "moa.classifiers.bayes.NaiveBayes");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "WindowClassificationPerformanceEvaluator");

    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public IntOption memCheckFrequencyOption = new IntOption(
            "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FileOption outputPredictionFileOption = new FileOption("outputPredictionFile", 'o',
            "File to append output predictions to.", null, "pred", true);

    //New for prequential method DEPRECATED
    public IntOption widthOption = new IntOption("width",
            'w', "Size of Window", 1000);

    public FloatOption alphaOption = new FloatOption("alpha",
            'a', "Fading factor or exponential smoothing factor", .01);

    public StringOption uuidOption = new StringOption("uuidPrefix", 'u',
            "uuidPrefix.", "");
    //End New for prequential methods

    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        Learner learner = (Learner) getPreparedClassOption(this.learnerOption);
        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);
        LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");

        //New for prequential methods
        if (evaluator instanceof WindowClassificationPerformanceEvaluator) {
            //((WindowClassificationPerformanceEvaluator) evaluator).setWindowWidth(widthOption.getValue());
            if (widthOption.getValue() != 1000) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (WindowClassificationPerformanceEvaluator -w " + widthOption.getValue() + ")");
                 return learningCurve;
            }
        }
        if (evaluator instanceof EWMAClassificationPerformanceEvaluator) {
            //((EWMAClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (EWMAClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        if (evaluator instanceof FadingFactorClassificationPerformanceEvaluator) {
            //((FadingFactorClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (FadingFactorClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        //End New for prequential methods

        learner.setModelContext(stream.getHeader());
        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);

        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        //File for output predictions
        File outputPredictionFile = this.outputPredictionFileOption.getFile();
        PrintStream outputPredictionResultStream = null;
        if (outputPredictionFile != null) {
            try {
                if (outputPredictionFile.exists()) {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile, true), true);
                } else {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFile, ex);
            }
        }
        boolean firstDump = true;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        //long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long evaluateStartTime = System.nanoTime();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;
        ArrayList<Double> measuredTimes = new ArrayList<>();
        ArrayList<Long> nInstances = new ArrayList<>();
        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
            Example trainInst = stream.nextInstance();


            Example testInst = (Example) trainInst; //.copy();
            //testInst.setClassMissing();
            double[] prediction = learner.getVotesForInstance(trainInst);
            // Output prediction
            if (outputPredictionFile != null) {
                int trueClass = (int) ((Instance) trainInst.getData()).classValue();
                outputPredictionResultStream.println(Utils.maxIndex(prediction) + "," + (
                 ((Instance) testInst.getData()).classIsMissing() == true ? " ? " : trueClass));
            }

            //evaluator.addClassificationAttempt(trueClass, prediction, testInst.weight());
            evaluator.addResult(testInst, prediction);
            learner.trainOnInstance(testInst);
            instancesProcessed++;
            if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                    || stream.hasMoreInstances() == false) {
                nInstances.add(instancesProcessed);
                //long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                long evaluateTime = System.nanoTime();
                measuredTimes.add(TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime));
                double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                //double RAMHoursIncrement = 0;
                double RAMHoursIncrement = learner.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                RAMHours += RAMHoursIncrement;
                lastEvaluateStartTime = evaluateTime;
                learningCurve.insertEntry(new LearningEvaluation(
                        new Measurement[]{
                            new Measurement(
                            "learning evaluation instances",
                            instancesProcessed),
                            new Measurement(
                            "evaluation time ("
                            + (preciseCPUTiming ? "cpu "
                            : "") + "seconds)",
                            time),
                            new Measurement(
                            "model cost (RAM-Hours)",
                            RAMHours)
                        },
                        evaluator, learner));
                if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }
            }
            if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                if (monitor.taskShouldAbort()) {
                    return null;
                }
                long estimatedRemainingInstances = stream.estimatedRemainingInstances();
                if (maxInstances > 0) {
                    long maxRemaining = maxInstances - instancesProcessed;
                    if ((estimatedRemainingInstances < 0)
                            || (maxRemaining < estimatedRemainingInstances)) {
                        estimatedRemainingInstances = maxRemaining;
                    }
                }
                monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                        : (double) instancesProcessed
                        / (double) (instancesProcessed + estimatedRemainingInstances));
                if (monitor.resultPreviewRequested()) {
                    monitor.setLatestResultPreview(learningCurve.copy());
                }
                //secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
                secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(System.nanoTime() - evaluateStartTime);
            }
        }
        learner.afterLearning();
        if (!uuidOption.getValue().equals("")) {
            Map<String, String> env = System.getenv();
            String dir = env.get("LOCAL_STORAGE_DIR") + "/Tmp/";
            String fileName = dir + "moaStatisticsRuntime_" + uuidOption.getValue() + ".csv";
            try {
                PrintWriter writer = new PrintWriter(new FileOutputStream(fileName, false));
                writer.println(Arrays.toString(nInstances.toArray()).replace("[", "").replace("]", ""));
                writer.println(Arrays.toString(measuredTimes.toArray()).replace("[", "").replace("]", ""));
                writer.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        if (outputPredictionResultStream != null) {
            outputPredictionResultStream.close();
        }
        return learningCurve;
    }
}
