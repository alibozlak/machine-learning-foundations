package dev.bozlak.regression.linear;

import java.util.ArrayList;

public class Utils {
    public static <T> int areInputsAndOutputsSameSize(ArrayList<T> inputs, ArrayList<Double> outputs)
        throws Exception {
        if (inputs.size() == outputs.size())
            return inputs.size();
        throw new Exception("Inputs size and Outputs size must be same!!");
    }
}
