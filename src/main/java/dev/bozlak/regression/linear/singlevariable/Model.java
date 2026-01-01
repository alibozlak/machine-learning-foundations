package dev.bozlak.regression.linear.singlevariable;

@FunctionalInterface
public interface Model {
    double function(double w, double b, double x);
}
