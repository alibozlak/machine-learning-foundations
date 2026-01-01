package dev.bozlak.regression.linear.multivariable;

@FunctionalInterface
public interface ModelForMultiVariableLinearRegression {
    double function(double[] weightVector, double b, double[] sampleInputVector);
}
