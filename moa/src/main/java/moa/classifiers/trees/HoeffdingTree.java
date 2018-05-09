/*
 *    HoeffdingTree.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
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
package moa.classifiers.trees;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import moa.AbstractMOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.InfoGainSplitCriterion;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.core.StringUtils;
import moa.core.Utils;
import moa.options.ClassOption;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import com.github.javacliparser.StringOption;
import java.util.Map;
import java.util.Random;
/**
 * Hoeffding Tree or VFDT.
 *
 * A Hoeffding tree is an incremental, anytime decision tree induction algorithm
 * that is capable of learning from massive data streams, assuming that the
 * distribution generating examples does not change over time. Hoeffding trees
 * exploit the fact that a small sample can often be enough to choose an optimal
 * splitting attribute. This idea is supported mathematically by the Hoeffding
 * bound, which quantiﬁes the number of observations (in our case, examples)
 * needed to estimate some statistics within a prescribed precision (in our
 * case, the goodness of an attribute).</p> <p>A theoretically appealing feature
 * of Hoeffding Trees not shared by other incremental decision tree learners is
 * that it has sound guarantees of performance. Using the Hoeffding bound one
 * can show that its output is asymptotically nearly identical to that of a
 * non-incremental learner using inﬁnitely many examples. See for details:</p>
 *
 * <p>G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
 * In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.</p>
 *
 * <p>Parameters:</p> <ul> <li> -m : Maximum memory consumed by the tree</li>
 * <li> -n : Numeric estimator to use : <ul> <li>Gaussian approximation
 * evaluating 10 splitpoints</li> <li>Gaussian approximation evaluating 100
 * splitpoints</li> <li>Greenwald-Khanna quantile summary with 10 tuples</li>
 * <li>Greenwald-Khanna quantile summary with 100 tuples</li>
 * <li>Greenwald-Khanna quantile summary with 1000 tuples</li> <li>VFML method
 * with 10 bins</li> <li>VFML method with 100 bins</li> <li>VFML method with
 * 1000 bins</li> <li>Exhaustive binary tree</li> </ul> </li> <li> -e : How many
 * instances between memory consumption checks</li> <li> -g : The number of
 * instances a leaf should observe between split attempts</li> <li> -s : Split
 * criterion to use. Example : InfoGainSplitCriterion</li> <li> -c : The
 * allowable error in split decision, values closer to 0 will take longer to
 * decide</li> <li> -t : Threshold below which a split will be forced to break
 * ties</li> <li> -b : Only allow binary splits</li> <li> -z : Stop growing as
 * soon as memory limit is hit</li> <li> -r : Disable poor attributes</li> <li>
 * -p : Disable pre-pruning</li> 
 *  <li> -l : Leaf prediction to use: MajorityClass (MC), Naive Bayes (NB) or NaiveBayes
 * adaptive (NBAdaptive).</li>
 *  <li> -q : The number of instances a leaf should observe before
 * permitting Naive Bayes</li>
 * </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */


public class HoeffdingTree extends AbstractClassifier {

    private static final long serialVersionUID = 1L;
    private static final int MAX_STEPS = 4000;
    private static final int MEASURE_GRACE_PERIOD = 5;
    private static final int INITIAL_GRACE_PERIOD = 100;
    private static int brentTotalIterations = 0;
    private static int brentSearches = 0;

    @Override
    public String getPurposeString() {
        return "Hoeffding Tree or VFDT.";
    }

    public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm',
            "Maximum memory consumed by the tree.", 33554432, 0,
            Integer.MAX_VALUE);

    /*
     * public MultiChoiceOption numericEstimatorOption = new MultiChoiceOption(
     * "numericEstimator", 'n', "Numeric estimator to use.", new String[]{
     * "GAUSS10", "GAUSS100", "GK10", "GK100", "GK1000", "VFML10", "VFML100",
     * "VFML1000", "BINTREE"}, new String[]{ "Gaussian approximation evaluating
     * 10 splitpoints", "Gaussian approximation evaluating 100 splitpoints",
     * "Greenwald-Khanna quantile summary with 10 tuples", "Greenwald-Khanna
     * quantile summary with 100 tuples", "Greenwald-Khanna quantile summary
     * with 1000 tuples", "VFML method with 10 bins", "VFML method with 100
     * bins", "VFML method with 1000 bins", "Exhaustive binary tree"}, 0);
     */
    public ClassOption numericEstimatorOption = new ClassOption("numericEstimator",
            'n', "Numeric estimator to use.", NumericAttributeClassObserver.class,
            "GaussianNumericAttributeClassObserver");

    public ClassOption nominalEstimatorOption = new ClassOption("nominalEstimator",
            'd', "Nominal estimator to use.", DiscreteAttributeClassObserver.class,
            "NominalAttributeClassObserver");

    public IntOption memoryEstimatePeriodOption = new IntOption(
            "memoryEstimatePeriod", 'e',
            "How many instances between memory consumption checks.", 1000000,
            0, Integer.MAX_VALUE);

    public IntOption gracePeriodOption = new IntOption(
            "gracePeriod",
            'g',
            "The number of instances a leaf should observe between split attempts.",
            200, 0, Integer.MAX_VALUE);


    public FlagOption measureSplitOffsetOption = new FlagOption("measureOffset", 'M',
            "Measure split offset.");

    public StringOption uuidOption = new StringOption("uuidPrefix", 'o',
            "uuidPrefix.", "");


    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            's', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");

    public FloatOption splitConfidenceOption = new FloatOption(
            "splitConfidence",
            'c',
            "The allowable error in split decision, values closer to 0 will take longer to decide.",
            0.0000001, 0.0, 1.0);

    public FloatOption tieThresholdOption = new FloatOption("tieThreshold",
            't', "Threshold below which a split will be forced to break ties.",
            0.05, 0.0, 1.0);

    public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
        "Only allow binary splits.");

    public FlagOption stopMemManagementOption = new FlagOption(
            "stopMemManagement", 'z',
            "Stop growing as soon as memory limit is hit.");

    public FlagOption removePoorAttsOption = new FlagOption("removePoorAtts",
            'r', "Disable poor attributes.");

    public FlagOption noPrePruneOption = new FlagOption("noPrePrune", 'p',
            "Disable pre-pruning.");

    public double trainOnInstanceTime=0.;
    public double attemptToSplitTime = 0;
    public double voteOnInstanceTime=0.;
    public int attempts=0;
    public int boundSplits=0;
    public int maxSplits=0;
    public int trainStepCount=0;
    public DoubleVector boundSplitErrors = new DoubleVector();
    public DoubleVector boundSplitNumSamples = new DoubleVector();
    public DoubleVector tieSplitNumSamples = new DoubleVector();

    public DoubleVector maxSplitErrors = new DoubleVector();
    private Random rand = new Random();

    public static class FoundNode {

        public Node node;

        public SplitNode parent;

        public int parentBranch;

        public FoundNode(Node node, SplitNode parent, int parentBranch) {
            this.node = node;
            this.parent = parent;
            this.parentBranch = parentBranch;
        }
    }

    public static class Node extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;

        protected DoubleVector observedClassDistribution;

        public Node(double[] classObservations) {
            //this.observedClassDistribution = new DoubleVector(classObservations);
            this.observedClassDistribution = new DoubleVector();
        }

        public int calcByteSize() {
            return (int) (SizeOf.sizeOf(this) + SizeOf.fullSizeOf(this.observedClassDistribution));
        }

        public int calcByteSizeIncludingSubtree() {
            return calcByteSize();
        }

        public boolean isLeaf() {
            return true;
        }

        public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                int parentBranch) {
            return new FoundNode(this, parent, parentBranch);
        }

        public double[] getObservedClassDistribution() {
            return this.observedClassDistribution.getArrayCopy();
        }

        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            return this.observedClassDistribution.getArrayCopy();
        }

        public boolean observedClassDistributionIsPure() {
            return this.observedClassDistribution.numNonZeroEntries() < 2;
        }

        public void describeSubtree(HoeffdingTree ht, StringBuilder out,
                int indent) {
            StringUtils.appendIndented(out, indent, "Leaf ");
            out.append(ht.getClassNameString());
            out.append(" = ");
            out.append(ht.getClassLabelString(this.observedClassDistribution.maxIndex()));
            out.append(" weights: ");
            this.observedClassDistribution.getSingleLineDescription(out,
                    ht.treeRoot.observedClassDistribution.numValues());
            StringUtils.appendNewline(out);
        }

        public int subtreeDepth() {
            return 0;
        }

        public double calculatePromise() {
            double totalSeen = this.observedClassDistribution.sumOfValues();
            return totalSeen > 0.0 ? (totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()))
                    : 0.0;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
            describeSubtree(null, sb, indent);
        }
    }

    public static class SplitNode extends Node {

        private static final long serialVersionUID = 1L;

        protected InstanceConditionalTest splitTest;

        protected AutoExpandVector<Node> children; // = new AutoExpandVector<Node>();

        @Override
        public int calcByteSize() {
            return super.calcByteSize()
                    + (int) (SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
        }

        @Override
        public int calcByteSizeIncludingSubtree() {
            int byteSize = calcByteSize();
            for (Node child : this.children) {
                if (child != null) {
                    byteSize += child.calcByteSizeIncludingSubtree();
                }
            }
            return byteSize;
        }

        public SplitNode(InstanceConditionalTest splitTest,
                double[] classObservations, int size) {
            super(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>(size);
        }
        
        public SplitNode(InstanceConditionalTest splitTest,
                double[] classObservations) {
            super(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>();
        }


        public int numChildren() {
            return this.children.size();
        }

        public void setChild(int index, Node child) {
            if ((this.splitTest.maxBranches() >= 0)
                    && (index >= this.splitTest.maxBranches())) {
                throw new IndexOutOfBoundsException();
            }
            this.children.set(index, child);
        }

        public Node getChild(int index) {
            return this.children.get(index);
        }

        public int instanceChildIndex(Instance inst) {
            return this.splitTest.branchForInstance(inst);
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        @Override
        public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                int parentBranch) {
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    return child.filterInstanceToLeaf(inst, this, childIndex);
                }
                return new FoundNode(null, this, childIndex);
            }
            return new FoundNode(this, parent, parentBranch);
        }

        @Override
        public void describeSubtree(HoeffdingTree ht, StringBuilder out,
                int indent) {
            for (int branch = 0; branch < numChildren(); branch++) {
                Node child = getChild(branch);
                if (child != null) {
                    StringUtils.appendIndented(out, indent, "if ");
                    out.append(this.splitTest.describeConditionForBranch(branch,
                            ht.getModelContext()));
                    out.append(": ");
                    StringUtils.appendNewline(out);
                    child.describeSubtree(ht, out, indent + 2);
                }
            }
        }

        @Override
        public int subtreeDepth() {
            int maxChildDepth = 0;
            for (Node child : this.children) {
                if (child != null) {
                    int depth = child.subtreeDepth();
                    if (depth > maxChildDepth) {
                        maxChildDepth = depth;
                    }
                }
            }
            return maxChildDepth + 1;
        }
    }

    public static abstract class LearningNode extends Node {

        private static final long serialVersionUID = 1L;

        public LearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public abstract void learnFromInstance(Instance inst, HoeffdingTree ht);
    }

    public static class InactiveLearningNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        public InactiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
        }
    }

    public static class ActiveLearningNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        protected double weightSeenAtLastSplitEvaluation;
        protected double initialWeight;

        protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<AttributeClassObserver>();
        
        protected boolean isInitialized;

        private int gracePeriod = 0;

        public ActiveLearningNode(double[] initialClassObservations, int gracePeriod) {
            this(initialClassObservations);
            this.gracePeriod = gracePeriod;
        }

        public ActiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.initialWeight = this.weightSeenAtLastSplitEvaluation;
            this.isInitialized = false;
        }

        @Override
        public int calcByteSize() {
            return super.calcByteSize()
                    + (int) (SizeOf.fullSizeOf(this.attributeObservers));
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            if (this.isInitialized == false) {
                this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
                this.isInitialized = true;
            }
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }

        public double getWeightSeen() {
            return this.observedClassDistribution.sumOfValues();
        }

        public double getWeightSeenAtLastSplitEvaluation() {
            return this.weightSeenAtLastSplitEvaluation;
        }

        public double getInitialWeight() {
            return this.initialWeight;
        }

        public void setWeightSeenAtLastSplitEvaluation(double weight) {
            this.weightSeenAtLastSplitEvaluation = weight;
        }

        public void setGracePeriod(int gracePeriod) {
            this.gracePeriod = gracePeriod;
        }

        public int getGracePeriod() {
            return this.gracePeriod;
        }

        public AttributeSplitSuggestion[] getBestSplitSuggestions(
                SplitCriterion criterion, HoeffdingTree ht) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
            double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
            if (!ht.noPrePruneOption.isSet()) {
                // add null split as an option
                bestSuggestions.add(new AttributeSplitSuggestion(null,
                        new double[0][], criterion.getMeritOfSplit(
                        preSplitDist,
                        new double[][]{preSplitDist})));
            }
            for (int i = 0; i < this.attributeObservers.size(); i++) {
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                            preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }

        public void disableAttribute(int attIndex) {
            this.attributeObservers.set(attIndex,
                    new NullAttributeClassObserver());
        }
    }

    protected Node treeRoot;

    protected int decisionNodeCount;

    protected int activeLeafNodeCount;

    protected int inactiveLeafNodeCount;

    protected double inactiveLeafByteSizeEstimate;

    protected double activeLeafByteSizeEstimate;

    protected double byteSizeEstimateOverheadFraction;

    protected boolean growthAllowed;

    public int calcByteSize() {
        int size = (int) SizeOf.sizeOf(this);
        if (this.treeRoot != null) {
            size += this.treeRoot.calcByteSizeIncludingSubtree();
        }
        return size;
    }

    @Override
    public int measureByteSize() {
        System.out.println("attemptTime " + this.attemptToSplitTime/1000. +  "s all " + (this.trainOnInstanceTime + this.voteOnInstanceTime)/1000. + "s");
        System.out.println("attempts " + this.attempts + " splits " + (this.boundSplits + this.maxSplits) + " boundsplits " + this.boundSplits + " maxsplits " + this.maxSplits);
        if (this.splitTimePredictionOption.getChosenIndex() == 2)
            System.out.println("Brent searchs " + brentSearches + " iterations " + brentTotalIterations);

        if (measureSplitOffsetOption.isSet())
            System.out.println("boundOffset " + Utils.mean(this.boundSplitErrors.getArrayRef()) + " maxOffset " + Utils.mean(this.maxSplitErrors.getArrayRef()));
        if (!uuidOption.getValue().equals("")) {
            Map<String, String> env = System.getenv();
            String dir = env.get("LOCAL_STORAGE_DIR") + "/Tmp/";
            try {
                String fileName = dir + "moaStatistics_" + uuidOption.getValue() + ".csv";
                PrintWriter writer = new PrintWriter(new FileOutputStream(fileName, false));
                writer.println(String.format("totalTime;attemptTime;attempts;boundSplits;maxSplits;boundOffset;maxOffset"));
                writer.println(String.format("%.2f;%.2f;%d;%d;%d;%.2f;%.2f", (this.trainOnInstanceTime + this.voteOnInstanceTime) / 1000., this.attemptToSplitTime / 1000., attempts,
                        boundSplits, maxSplits, Utils.mean(this.boundSplitErrors.getArrayRef()), Utils.mean(this.maxSplitErrors.getArrayRef())));
                writer.close();

                fileName = dir + "moaSplitNumSamples_" + uuidOption.getValue() + ".csv";
                writer = new PrintWriter(new FileOutputStream(fileName, false));
                writer.println(Utils.arrayToString(this.boundSplitNumSamples.getArrayRef()));
                writer.println(Utils.arrayToString(this.tieSplitNumSamples.getArrayRef()));
                writer.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        return calcByteSize();
    }

    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        this.inactiveLeafByteSizeEstimate = 0.0;
        this.activeLeafByteSizeEstimate = 0.0;
        this.byteSizeEstimateOverheadFraction = 1.0;
        this.growthAllowed = true;
        if (this.leafpredictionOption.getChosenIndex()>0) { 
            this.removePoorAttsOption = null;
        }
    }

    public void singleFit(LearningNode learningNode, SplitNode parent, int parentBranch){
        if (this.growthAllowed
                && (learningNode instanceof ActiveLearningNode)) {
            ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
            double weightSeen = activeLearningNode.getWeightSeen();
            int gracePeriod = this.gracePeriodOption.getValue();
            if (this.splitTimePredictionOption.getChosenIndex() > 0 && activeLearningNode.getGracePeriod() > 0) {
                gracePeriod = activeLearningNode.getGracePeriod();
            }
            if (weightSeen
                    - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= gracePeriod) {

                if (!activeLearningNode.observedClassDistributionIsPure()) {
                    this.attempts++;
                    attemptToSplit(activeLearningNode, parent, parentBranch, true);
                }
                activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
            }
        }
    }

    public void singleFitMeasure(LearningNode learningNode, SplitNode parent, int parentBranch){
        if (this.growthAllowed
                && (learningNode instanceof ActiveLearningNode)) {
            ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
            double weightSeen = activeLearningNode.getWeightSeen();

            //XXVL here!
            int gracePeriod = HoeffdingTree.MEASURE_GRACE_PERIOD;
            boolean adaptGracePeriod = true;
            if (activeLearningNode.getWeightSeenAtLastSplitEvaluation() == activeLearningNode.getInitialWeight()){
                gracePeriod = HoeffdingTree.INITIAL_GRACE_PERIOD;
                adaptGracePeriod = false;
            }
            else if (weightSeen < activeLearningNode.getInitialWeight() + this.gracePeriodOption.getValue()) {
                adaptGracePeriod = false;
            }
            else {
                //adaptGracePeriod = true;
                //gracePeriod = HoeffdingTree.MEASURE_GRACE_PERIOD;
            }

            if (weightSeen
                    - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= gracePeriod) {
                activeLearningNode.setGracePeriod(activeLearningNode.getGracePeriod() - gracePeriod);
                if (!activeLearningNode.observedClassDistributionIsPure()) {
                    int splitReturn;
                    if (activeLearningNode.getGracePeriod() <= 0) {
                        splitReturn = attemptToSplit(activeLearningNode, parent, parentBranch, adaptGracePeriod);
                        this.attempts++;

                    } else {
                        splitReturn = attemptToSplit(activeLearningNode, parent, parentBranch, false);
                    }
                    if (splitReturn == 1){
                        this.boundSplitErrors.addToValue(this.boundSplitErrors.numValues(),Math.max(0., activeLearningNode.getGracePeriod()));
                    }

                    else if (splitReturn == 2) {
                        this.maxSplitErrors.addToValue(this.maxSplitErrors.numValues(), Math.max(0., activeLearningNode.getGracePeriod()));
                    }
                }
                activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainStepCount++;
        long startTime = System.currentTimeMillis();
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }

        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (measureSplitOffsetOption.isSet())
                this.singleFitMeasure(learningNode, foundNode.parent, foundNode.parentBranch);
            else
                this.singleFit(learningNode, foundNode.parent, foundNode.parentBranch);
        }

        if (this.trainingWeightSeenByModel
                % this.memoryEstimatePeriodOption.getValue() == 0) {
            estimateModelByteSizes();
        }
        this.trainOnInstanceTime += System.currentTimeMillis() - startTime;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        long startTime = System.currentTimeMillis();
        if (this.treeRoot != null) {
            FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst,
                    null, -1);
            Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            this.voteOnInstanceTime += System.currentTimeMillis() - startTime;
            return leafNode.getClassVotes(inst, this);
          } else {
            int numClasses = inst.dataset().numClasses();
            this.voteOnInstanceTime += System.currentTimeMillis() - startTime;
            return new double[numClasses];
          }

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
                    new Measurement("tree size (nodes)", this.decisionNodeCount
                    + this.activeLeafNodeCount + this.inactiveLeafNodeCount),
                    new Measurement("tree size (leaves)", this.activeLeafNodeCount
                    + this.inactiveLeafNodeCount),
                    new Measurement("active learning leaves",
                    this.activeLeafNodeCount),
                    new Measurement("tree depth", measureTreeDepth()),
                    new Measurement("active leaf byte size estimate",
                    this.activeLeafByteSizeEstimate),
                    new Measurement("inactive leaf byte size estimate",
                    this.inactiveLeafByteSizeEstimate),
                    new Measurement("byte size estimate overhead",
                    this.byteSizeEstimateOverheadFraction)};
    }

    public int measureTreeDepth() {
        if (this.treeRoot != null) {
            return this.treeRoot.subtreeDepth();
        }
        return 0;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        this.treeRoot.describeSubtree(this, out, indent);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    public static double computeHoeffdingBound(double range, double confidence,
            double n) {
        return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
    }

    //Procedure added for Hoeffding Adaptive Trees (ADWIN)
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
            double[] classObservations, int size) {
        return new SplitNode(splitTest, classObservations, size);
    }
    
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
            double[] classObservations) {
        return new SplitNode(splitTest, classObservations);
    }
    

    protected AttributeClassObserver newNominalClassObserver() {
        AttributeClassObserver nominalClassObserver = (AttributeClassObserver) getPreparedClassOption(this.nominalEstimatorOption);
        return (AttributeClassObserver) nominalClassObserver.copy();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        AttributeClassObserver numericClassObserver = (AttributeClassObserver) getPreparedClassOption(this.numericEstimatorOption);
        return (AttributeClassObserver) numericClassObserver.copy();
    }

    protected void splitNode(ActiveLearningNode node, SplitNode parent, int parentIndex, AttributeSplitSuggestion splitDecision){
        if (splitDecision.splitTest == null) {
            // preprune - null wins
            deactivateLearningNode(node, parent, parentIndex);
        } else {
            SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                    node.getObservedClassDistribution(),splitDecision.numSplits() );
            for (int i = 0; i < splitDecision.numSplits(); i++) {
                Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                newSplit.setChild(i, newChild);
            }
            this.activeLeafNodeCount--;
            this.decisionNodeCount++;
            this.activeLeafNodeCount += splitDecision.numSplits();
            if (parent == null) {
                this.treeRoot = newSplit;
            } else {
                parent.setChild(parentIndex, newSplit);
            }
        }
        // manage memory
        enforceTrackerLimit();
    }

    public static double getHoeffdingN(double range, double confidence, double delta) {
        return ((range * range) * Math.log(1.0 / confidence)) / (2.0 * Math.pow(delta, 2.));
    }

    static double Brent(double a,  double b, double iEps, double oEps, IOptFunction optFunction, Object...params)
    {
        double  fa, fb, fc, fs, c, c0, c1, c2,temp, d, s;
        int mflag;

        c = a;
        d = c;
        fa = optFunction.func(a, params);
        fb = optFunction.func(b, params);
        fc = fa;

        if (fa*fb >= 0) {
            System.out.println("Error no root between x0 and x1: f(x0) = " + fa + " f(x1)=" + fb);
            return -1.;
        }

        if ( Math.abs(fa) < Math.abs(fb))
        {
            temp = a;
            a = b;
            b = temp;
            temp = fa;
            fa = fb;
            fb = temp;
        } // if

        mflag = 1;
        brentSearches++;
        while ( (Math.abs(fb) > oEps) && ( Math.abs(b-a) > iEps))
        {
            brentTotalIterations++;
            if ( (fa != fc) && fb != fc)
            {
                c0 = a*fb*fc/((fa-fb)*(fa-fc));
                c1 = b*fa*fc/((fb-fa)*(fb-fc));
                c2 = c*fa*fb/((fc-fa)*(fc-fb));

                s = c0 + c1 + c2;
            } // if
            else
                s = b - fb*(b-a)/(fb - fa);

            if ( ( s < (3*(a+b)/4) || s > b) || ( (mflag == 1) &&
                    Math.abs(s-b) >= (Math.abs(b-c)/2) ) ||
                    ( (mflag == 0) && Math.abs(s-b) >= (Math.abs(c-d)/2) ) )
            {
                s = (a+b)/2;
                mflag = 1;
            } //if
            else
                mflag = 0;

            fs = optFunction.func(s, params);
            d = c;
            c = b;
            fc = fb;
            if ( (fa*fs)< 0)
                b = s;
            else
                a = s;
            if ( Math.abs(fa) < Math.abs(fb))
            {
                temp = a;
                a = b;
                b = temp;
                temp = fa;
                fa = fb;
                fb = temp;
            } // if

        } // while

        return b;
    } //brent

    interface IOptFunction {
        double func(double x, Object[] params);
    }

    static class OptFunction implements IOptFunction {
        public double func(double x, Object[] params) {
            double[][] dists = ((double[][])params[0]).clone();
            int minEntropyIdx = (int)params[1];
            int maxClassIdx = (int)params[2];
            double secEntropy = (double)params[3];
            double entropyRange = (double)params[4];
            double splitConfidence = (double)params[5];
            double numTrainSamples = (double)params[6];

            double delta = computeHoeffdingBound(entropyRange, splitConfidence, numTrainSamples + x);
            dists[minEntropyIdx] = dists[minEntropyIdx].clone();
            dists[minEntropyIdx][maxClassIdx] += x;
            double entropy = InfoGainSplitCriterion.computeEntropy(dists);
            double result = secEntropy - entropy - delta;
            return result;
        }
    }

    public static int splitTimePredictionCGD(AttributeSplitSuggestion bestSuggestion, AttributeSplitSuggestion secondBestSuggestion, double entropyRange, double splitConfidence, double numTrainSamples, double tieDelta, int gracePeriod){
        double delta = bestSuggestion.merit - secondBestSuggestion.merit;
        if (delta > 0) {
            double estimate = HoeffdingTree.getHoeffdingN(entropyRange, splitConfidence, delta) - numTrainSamples;
            return (int) Math.ceil(Math.min(estimate, HoeffdingTree.MAX_STEPS));
        } else
            return HoeffdingTree.MAX_STEPS;

    }

    public static int splitTimePredictionOSM(AttributeSplitSuggestion bestSuggestion, AttributeSplitSuggestion secondBestSuggestion, double entropyRange, double splitConfidence, double numTrainSamples, double tieDelta, int gracePeriod){
        double minEntropy = Double.MAX_VALUE;
        int minEntropyIdx = 0;
        for (int i = 0; i < bestSuggestion.resultingClassDistributions.length; i++){
            double weight = Utils.sum(bestSuggestion.resultingClassDistributions[i]);
            double entropy = InfoGainSplitCriterion.computeEntropy(bestSuggestion.resultingClassDistributions[i]);
            if (entropy < minEntropy && weight > 0){
                minEntropyIdx = i;
                minEntropy = entropy;
            }
        }
        double secEntropy = InfoGainSplitCriterion.computeEntropy(secondBestSuggestion.resultingClassDistributions);

        double[][] dists = bestSuggestion.resultingClassDistributions.clone();
        int maxClassIdx = Utils.maxIndex(dists[minEntropyIdx]);
        double delta = computeHoeffdingBound(entropyRange, splitConfidence, numTrainSamples + HoeffdingTree.MAX_STEPS);
        dists[minEntropyIdx] = dists[minEntropyIdx].clone();

        dists[minEntropyIdx][maxClassIdx] += HoeffdingTree.MAX_STEPS;
        double tmpEntropy = InfoGainSplitCriterion.computeEntropy(dists);
        double result = HoeffdingTree.MAX_STEPS;

        if (secEntropy - tmpEntropy > delta){
            result = HoeffdingTree.Brent(0, (double)HoeffdingTree.MAX_STEPS, 0.5, 0.005, new OptFunction(),
                    bestSuggestion.resultingClassDistributions, minEntropyIdx, maxClassIdx, secEntropy, entropyRange, splitConfidence, numTrainSamples);
        }
        result = Math.ceil(Math.min(Math.max(gracePeriod, result), HoeffdingTree.MAX_STEPS));
        return (int)result;
    }

    protected int attemptToSplit(ActiveLearningNode node, SplitNode parent, int parentIndex, boolean adaptGracePeriod) {
        double startTime = System.currentTimeMillis();

        SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
        AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
        Arrays.sort(bestSplitSuggestions);
        int splitResult = 0;
        if (bestSplitSuggestions.length == 1) {
            this.boundSplits++;
            this.splitNode(node, parent, parentIndex, bestSplitSuggestions[bestSplitSuggestions.length - 1]);
            splitResult = 1;
        } else if (bestSplitSuggestions.length >= 2) {
            double criterionRange = splitCriterion.getRangeOfMerit(node.getObservedClassDistribution());
            double hoeffdingBound = computeHoeffdingBound(criterionRange,
                    this.splitConfidenceOption.getValue(), node.getWeightSeen());
            AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
            AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
            if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                    || hoeffdingBound < this.tieThresholdOption.getValue()) {
                if (hoeffdingBound < this.tieThresholdOption.getValue()) {
                    this.maxSplits++;
                    splitResult = 2;
                    this.tieSplitNumSamples.addToValue(this.tieSplitNumSamples.numValues(), node.getWeightSeen() - node.initialWeight);
                }
                else {
                    this.boundSplits++;
                    this.boundSplitNumSamples.addToValue(this.boundSplitNumSamples.numValues(), node.getWeightSeen() - node.initialWeight);
                    splitResult = 1;
                }


                this.splitNode(node, parent, parentIndex, bestSuggestion);
            } else {
                if (adaptGracePeriod) {
                    double maxN = HoeffdingTree.getHoeffdingN(criterionRange, splitConfidenceOption.getValue(), tieThresholdOption.getValue()) - node.getWeightSeen();
                    double gracePeriod = this.gracePeriodOption.getValue();
                    if (this.splitTimePredictionOption.getChosenIndex() == 1) {
                        gracePeriod = HoeffdingTree.splitTimePredictionCGD(bestSuggestion, secondBestSuggestion, criterionRange, splitConfidenceOption.getValue(), node.getWeightSeen(), tieThresholdOption.getValue(), gracePeriodOption.getValue());
                    } else if (this.splitTimePredictionOption.getChosenIndex() == 2) {
                        gracePeriod = HoeffdingTree.splitTimePredictionOSM(bestSuggestion, secondBestSuggestion, criterionRange, splitConfidenceOption.getValue(), node.getWeightSeen(), tieThresholdOption.getValue(), gracePeriodOption.getValue());
                    } else if (this.splitTimePredictionOption.getChosenIndex() == 3) {
                        gracePeriod = rand.nextInt(MAX_STEPS - this.gracePeriodOption.getValue() + 1) + this.gracePeriodOption.getValue();
                    }
                    node.gracePeriod = (int) Math.ceil(Math.min(maxN, gracePeriod));
                }

                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2wwwwe - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }
        }
        this.attemptToSplitTime += System.currentTimeMillis() - startTime;
        return splitResult;
    }

    public void enforceTrackerLimit() {
        if ((this.inactiveLeafNodeCount > 0)
                || ((this.activeLeafNodeCount * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate)
                * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue())) {
            if (this.stopMemManagementOption.isSet()) {
                this.growthAllowed = false;
                return;
            }
            FoundNode[] learningNodes = findLearningNodes();
            Arrays.sort(learningNodes, new Comparator<FoundNode>() {

                @Override
                public int compare(FoundNode fn1, FoundNode fn2) {
                    return Double.compare(fn1.node.calculatePromise(), fn2.node.calculatePromise());
                }
            });
            int maxActive = 0;
            while (maxActive < learningNodes.length) {
                maxActive++;
                if ((maxActive * this.activeLeafByteSizeEstimate + (learningNodes.length - maxActive)
                        * this.inactiveLeafByteSizeEstimate)
                        * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue()) {
                    maxActive--;
                    break;
                }
            }
            int cutoff = learningNodes.length - maxActive;
            for (int i = 0; i < cutoff; i++) {
                if (learningNodes[i].node instanceof ActiveLearningNode) {
                    deactivateLearningNode(
                            (ActiveLearningNode) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
            for (int i = cutoff; i < learningNodes.length; i++) {
                if (learningNodes[i].node instanceof InactiveLearningNode) {
                    activateLearningNode(
                            (InactiveLearningNode) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
        }
    }

    public void estimateModelByteSizes() {
        FoundNode[] learningNodes = findLearningNodes();
        long totalActiveSize = 0;
        long totalInactiveSize = 0;
        for (FoundNode foundNode : learningNodes) {
            if (foundNode.node instanceof ActiveLearningNode) {
                totalActiveSize += SizeOf.fullSizeOf(foundNode.node);
            } else {
                totalInactiveSize += SizeOf.fullSizeOf(foundNode.node);
            }
        }
        if (totalActiveSize > 0) {
            this.activeLeafByteSizeEstimate = (double) totalActiveSize
                    / this.activeLeafNodeCount;
        }
        if (totalInactiveSize > 0) {
            this.inactiveLeafByteSizeEstimate = (double) totalInactiveSize
                    / this.inactiveLeafNodeCount;
        }
        int actualModelSize = this.measureByteSize();
        double estimatedModelSize = (this.activeLeafNodeCount
                * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate);
        this.byteSizeEstimateOverheadFraction = actualModelSize
                / estimatedModelSize;
        if (actualModelSize > this.maxByteSizeOption.getValue()) {
            enforceTrackerLimit();
        }
    }

    public void deactivateAllLeaves() {
        FoundNode[] learningNodes = findLearningNodes();
        for (int i = 0; i < learningNodes.length; i++) {
            if (learningNodes[i].node instanceof ActiveLearningNode) {
                deactivateLearningNode(
                        (ActiveLearningNode) learningNodes[i].node,
                        learningNodes[i].parent, learningNodes[i].parentBranch);
            }
        }
    }

    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
            SplitNode parent, int parentBranch) {
        Node newLeaf = new InactiveLearningNode(toDeactivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount--;
        this.inactiveLeafNodeCount++;
    }

    protected void activateLearningNode(InactiveLearningNode toActivate,
            SplitNode parent, int parentBranch) {
        Node newLeaf = newLearningNode(toActivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount++;
        this.inactiveLeafNodeCount--;
    }

    protected FoundNode[] findLearningNodes() {
        List<FoundNode> foundList = new LinkedList<FoundNode>();
        findLearningNodes(this.treeRoot, null, -1, foundList);
        return foundList.toArray(new FoundNode[foundList.size()]);
    }

    protected void findLearningNodes(Node node, SplitNode parent,
            int parentBranch, List<FoundNode> found) {
        if (node != null) {
            if (node instanceof LearningNode) {
                found.add(new FoundNode(node, parent, parentBranch));
            }
            if (node instanceof SplitNode) {
                SplitNode splitNode = (SplitNode) node;
                for (int i = 0; i < splitNode.numChildren(); i++) {
                    findLearningNodes(splitNode.getChild(i), splitNode, i,
                            found);
                }
            }
        }
    }

    public MultiChoiceOption leafpredictionOption = new MultiChoiceOption(
            "leafprediction", 'l', "Leaf prediction to use.", new String[]{
                "MC", "NB", "NBAdaptive"}, new String[]{
                "Majority class",
                "Naive Bayes",
                "Naive Bayes Adaptive"}, 2);

    public MultiChoiceOption splitTimePredictionOption = new MultiChoiceOption(
            "splitTimePrediction", 'a', "Grace period.", new String[]{
            "None", "CGD", "OSM", "RND"}, new String[]{
            "No prediction",
            "Constant gain difference",
            "One sided minimum",
            "Random"}, 0);

    public IntOption nbThresholdOption = new IntOption(
            "nbThreshold",
            'q',
            "The number of instances a leaf should observe before permitting Naive Bayes.",
            0, 0, Integer.MAX_VALUE);

    public static class LearningNodeNB extends ActiveLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public LearningNodeNB(double[] initialClassObservations, int gracePeriod) {
            super(initialClassObservations, gracePeriod);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
                return NaiveBayes.doNaiveBayesPrediction(inst,
                        this.observedClassDistribution,
                        this.attributeObservers);
            }
            return super.getClassVotes(inst, ht);
        }

        @Override
        public void disableAttribute(int attIndex) {
            // should not disable poor atts - they are used in NB calc
        }
    }

    public static class LearningNodeNBAdaptive extends LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptive(double[] initialClassObservations, int adaptiveGracePeriod) {
            super(initialClassObservations, adaptiveGracePeriod);
        }

        public LearningNodeNBAdaptive(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            int trueClass = (int) inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }
            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }
            super.learnFromInstance(inst, ht);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    protected LearningNode newLearningNode() {
        return newLearningNode(new double[0]);
    }

    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new ActiveLearningNode(initialClassObservations, this.gracePeriodOption.getValue());
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations, this.gracePeriodOption.getValue());
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations, this.gracePeriodOption.getValue());
        }
        return ret;
    }
}
