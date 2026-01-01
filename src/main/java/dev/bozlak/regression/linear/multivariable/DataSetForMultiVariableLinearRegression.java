package dev.bozlak.regression.linear.multivariable;

import dev.bozlak.regression.linear.Utils;

import java.util.ArrayList;
import java.util.Arrays;

public class DataSetForMultiVariableLinearRegression {
    private final ArrayList<double[]> inputs;
    private final ArrayList<Double> outputs;
    private int m;
    public final ModelForMultiVariableLinearRegression modelForMultiVariableLinearRegression
            = (weightVector, b, sampleInputVector) -> {
      double result = 0;
      for (int i = 0; i < weightVector.length; i++){
          result += (weightVector[i] * sampleInputVector[i]);
      }
      return result + b;
    };

    public DataSetForMultiVariableLinearRegression(ArrayList<double[]> inputs, ArrayList<Double> outputs)
        throws Exception {
        this.m = Utils.areInputsAndOutputsSameSize(inputs,outputs);
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public int getM(){
        return this.m;
    }

    public int getDataSetCount(){
        return this.m;
    }

    public double J(double[] weightVector, double b){
        double j = 0;
        for (int i = 0; i < this.m; i++){
             j += Math.pow(this.modelForMultiVariableLinearRegression.function(weightVector,b,inputs.get(i)) -
                    this.outputs.get(i), 2);
        }
        return j / (2 * this.m);
    }

    private double dJ_dwj(int jThIndexOfWeightVector, double[] weightVector, double b){
        double result = 0;
        for (int i = 0; i < this.m; i++){
            result += this.inputs.get(i)[jThIndexOfWeightVector] *
                    (this.modelForMultiVariableLinearRegression.function(weightVector, b, this.inputs.get(i)) -
                            this.outputs.get(i));
        }
        return result / this.m;
    }

    private double dJ_db(double[] weightVector, double b){
        double result = 0;
        for (int i = 0; i < this.m; i++){
            result += (this.modelForMultiVariableLinearRegression.function(weightVector, b, this.inputs.get(i))
                - this.outputs.get(i));
        }
        return result / this.m;
    }

    public WeightVectorAndBias trainDataSet(int iterationCount, double learningRate, double[] initWeightVector, double b){
        WeightVectorAndBias weightVectorAndBias = new WeightVectorAndBias(initWeightVector, b);
        double[] weightVector = weightVectorAndBias.weightVector;
        for (int i = 0; i < iterationCount; i++){
            double[] tempWeightVector = Arrays.copyOf(weightVector,weightVector.length);
            double bias = weightVectorAndBias.b;
            for (int j = 0; j < initWeightVector.length; j++){
                weightVector[j] -= learningRate * dJ_dwj(j, tempWeightVector, bias);
            }
            weightVectorAndBias.b -= learningRate * dJ_db(tempWeightVector, bias);
        }
        return weightVectorAndBias;
    }


    public static class WeightVectorAndBias {
        public double[] weightVector;
        public double b;

        public WeightVectorAndBias(double[] weightVector, double b){
            this.weightVector = weightVector;
            this.b = b;
        }
    }
}
