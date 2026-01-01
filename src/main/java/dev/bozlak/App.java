package dev.bozlak;

import dev.bozlak.regression.linear.multivariable.DataSetForMultiVariableLinearRegression;
import dev.bozlak.regression.linear.singlevariable.DataSetForSingleVariableLinearRegression;

import java.util.ArrayList;
import java.util.Arrays;

public class App
{
    public static void main( String[] args ) throws Exception
    {
        //testOfSingleVariableLinearRegression();
        //testForMultivariableLinearRegression();
        //test2ForMultivariableLinearRegression();
    }

    private static void test2ForMultivariableLinearRegression() throws Exception {
        // Eğitim Verileri (Mutfaktan gelen gerçek veriler)
        ArrayList<double[]> inputs = new ArrayList<>();
        inputs.add(new double[]{50, 1});  // 50m2, 1 oda
        inputs.add(new double[]{100, 3}); // 100m2, 3 oda
        inputs.add(new double[]{150, 4}); // 150m2, 4 oda
        //inputs.add(new double[]{200, 5}); // 200m2, 5 oda

        ArrayList<Double> outputs = new ArrayList<>();
        outputs.add(10.0); // 1.000.000 TL
        outputs.add(21.0); // 2.100.000 TL
        outputs.add(31.0); // 3.100.000 TL
        //outputs.add(41.0); // 4.100.000 TL

        DataSetForMultiVariableLinearRegression ds = new DataSetForMultiVariableLinearRegression(inputs, outputs);

        // Parametreler (Dikkat: metrekareler büyük olduğu için learningRate'i çok küçük tutmalısın)
        int iterations = 100000;
        double alpha = 0.0001;
        double[] initialW = {0.0, 0.0};
        double initialB = 0.0;

        // EĞİTİM
        DataSetForMultiVariableLinearRegression.WeightVectorAndBias trainedModel =
                ds.trainDataSet(iterations, alpha, initialW, initialB);

        System.out.println("Weights [w1,w1] = " + Arrays.toString(trainedModel.weightVector));
        //Weights [w1,w1] = [0.19589967646126527, 0.4683945481180375]
        System.out.println("b = " + trainedModel.b);
        //b = -0.1715369758487028
        System.out.println("Latest J = " + ds.J(trainedModel.weightVector, trainedModel.b));
        //Latest J = 0.007852413214066612

        //new real data
        double[] houseM2andRoomCount = new double[]{200, 5}; // 200m2, 5 oda
        double housePrice = 41.0;  // 4.100.000 TL

        System.out.println("Trained Model Predict of new real data : "
                + ds.modelForMultiVariableLinearRegression.function(
                        trainedModel.weightVector, trainedModel.b, houseM2andRoomCount));
        System.out.println("Real price : " + housePrice);
    }

    private static void testForMultivariableLinearRegression() throws Exception {
        // 1. Veri setini hazırla
        ArrayList<double[]> inputs = new ArrayList<>();
        // [m2, oda_sayisi]
        inputs.add(new double[]{60.0, 2.0});
        inputs.add(new double[]{80.0, 3.0});
        inputs.add(new double[]{100.0, 3.0});
        inputs.add(new double[]{120.0, 4.0});

        ArrayList<Double> outputs = new ArrayList<>();
        // Fiyat (milyon TL)
        outputs.add(1.5);
        outputs.add(2.1);
        outputs.add(2.4);
        outputs.add(3.2);

        // 2. Modeli oluştur
        DataSetForMultiVariableLinearRegression multiModel =
                new DataSetForMultiVariableLinearRegression(inputs, outputs);

        // 3. Eğitimi başlat
        int iteration = 1000;
        double alpha = 0.0001; // Metrekare değerleri büyük olduğu için düşük bir learning rate seçtik
        double[] initialWeights = new double[]{0.0, 0.0};
        double initialBias = 0.0;

        DataSetForMultiVariableLinearRegression.WeightVectorAndBias result =
                multiModel.trainDataSet(iteration, alpha, initialWeights, initialBias);

        // 4. Sonuçları yazdır
        System.out.println("Ağırlıklar (w1, w2): " + Arrays.toString(result.weightVector));
        System.out.println("Sapma (b): " + result.b);
        System.out.println("Son Hata (J): " + multiModel.J(result.weightVector, result.b));
        //f(x1, x2) = 0.025558205092659648 * x1 + 0.0026108895841007018 * x2 + -4.907117163342739E-4
    }


    private static void testOfSingleVariableLinearRegression() throws Exception {
        ArrayList<Double> inputs = new ArrayList<>();
        inputs.add(0.7);
        inputs.add(1.0);
        inputs.add(2.);
        inputs.add(2.5);
        inputs.add(3.);

        ArrayList<Double> labels = new ArrayList<>();
        labels.add(3.);
        labels.add(1.);
        labels.add(3.);
        labels.add(6.);
        labels.add(5.6);

        DataSetForSingleVariableLinearRegression dataSetForSingleVariableLinearRegression =
                new DataSetForSingleVariableLinearRegression(inputs,labels);
        int iterationCount = 1_000;
        double learningRate = 0.01;
        DataSetForSingleVariableLinearRegression.WeightAndBias weightAndBias =
                dataSetForSingleVariableLinearRegression.trainDataSet(iterationCount,learningRate,0,0);
        System.out.println("Result w : " + weightAndBias.weight);
        System.out.println("Result b : " + weightAndBias.bias);
        double J = dataSetForSingleVariableLinearRegression.J(weightAndBias.weight, weightAndBias.bias);
        System.out.println("J : " + J);

        // iterationCount = 100, learningRate = 0.5, initW = 0, initB = 0 -> J = 9.815083007616678E35
        // iterationCount = 1000, learningRate = 0.5, initW = 0, initB = 0 -> J = Infinity

        // iterationCount = 1000, learningRate = 0.01, initW = 0, initB = 0 -> J = 0.5479075875256325
        // iterationCount = 10_000, learningRate = 0.01, initW = 0, initB = 0 -> J = 0.5476243441762879
    }
}
