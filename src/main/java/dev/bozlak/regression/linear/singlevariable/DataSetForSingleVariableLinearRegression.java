package dev.bozlak.regression.linear.singlevariable;

import java.util.ArrayList;

public class SingleVariableLinearRegression {
    private ArrayList<Double> inputs;
    private ArrayList<Double> labels;
    private int m;
    public Model function = (w, b, x) -> w * x + b;

    public SingleVariableLinearRegression(ArrayList<Double> inputs, ArrayList<Double> labels) throws Exception {
        this.areInputsAndLabelsSameSize(inputs, labels);
    }

    public SingleVariableLinearRegression(ArrayList<Double> inputs, ArrayList<Double> labels, Model function)
            throws Exception {
        this.areInputsAndLabelsSameSize(inputs, labels);
        this.function = function;
    }

    public ArrayList<Double> getInputs() {
        return inputs;
    }

    public ArrayList<Double> getLabels() {
        return labels;
    }

    public int getM(){
        return this.m;
    }

    public int getDataSetSize(){
        return this.m;
    }

    private void areInputsAndLabelsSameSize(ArrayList<Double> inputs, ArrayList<Double> labels) throws Exception {
        int inputSize = inputs.size();
        if (inputSize == labels.size()){
            this.inputs = inputs;
            this.labels = labels;
            this.m = inputSize;
        } else {
            throw new Exception("Inputs and Labels must be same size!!");
        }
    }

    public double J(double w, double b){
        double result = 0;
        for (int i = 0; i < this.m; i++){
            result += Math.pow(this.function.function(w, b, inputs.get(i)) - labels.get(i), 2);
        }
        return result / (2 * this.m);
    }

    public static class WeightAndBias {
        public double weight;
        public double bias;

        public WeightAndBias(double weight, double bias){
            this.weight = weight;
            this.bias = bias;
        }

        public WeightAndBias() {
        }
    }

    private double dJ_dw(double w, double b){
        double result = 0;
        for (int i = 0; i < this.m; i++){
            double iThInputValue = this.inputs.get(i);
            result += iThInputValue * (this.function.function(w,b,iThInputValue) - this.labels.get(i));
        }
        return result / this.m;
    }

    private double dJ_db(double w, double b){
        double result = 0;
        for (int i = 0; i < this.m; i++){
            result += (this.function.function(w,b,this.inputs.get(i)) - this.labels.get(i));
        }
        return result / this.m;
    }

    public WeightAndBias trainDataSet(int iterationCount, double learningRate, double initW, double initB){
        WeightAndBias weightAndBias = new WeightAndBias(initW,initB);
        for (int i = 0; i < iterationCount; i++){
            double weight = weightAndBias.weight;
            double bias = weightAndBias.bias;
            weightAndBias.weight = weight - learningRate * this.dJ_dw(weight,bias);
            weightAndBias.bias = bias - learningRate * this.dJ_db(weight,bias);
        }
        return weightAndBias;
    }
}
