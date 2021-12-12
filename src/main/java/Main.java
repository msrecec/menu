import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

import java.util.Scanner;

public class Main {

    public static double Input[][];
    public static double Ideal[][];
    public static String idealno[] = { "piletina", "sok", "kava", "vino", "svinjetina" };

    public static int naj = 9;

    public static void main(String[] args) {
        double l[] = new double[idealno.length + 1];
        for (int i = 1; i < idealno.length + 1; i++) {
            l[i] = 1.0 / idealno.length * i;
        }

        Input = new double[][] { norm("kylling"), norm("kyckling"), norm("chicken"), norm("poulet"), norm("juice"), norm("juice"),
                norm("juice"), norm("jus"), norm("kaffe"), norm("kaffe"), norm("coffee"), norm("cafe"), norm("vin"),
                norm("vin"), norm("wine"), norm("vin"), norm("svinekod"), norm("flask"), norm("pork"), norm("porc") };

        Ideal = new double[][] { { l[1] }, { l[1] }, { l[1] }, { l[1] }, { l[2] }, { l[2] }, { l[2] }, { l[2] }, { l[3] },
                { l[3] }, { l[3] }, { l[3] }, { l[4] }, { l[4] }, { l[4] }, { l[4] }, { l[5] }, { l[5] }, { l[5] }, { l[5] } };

        BasicNetwork net = new BasicNetwork();
        net.addLayer(new BasicLayer(null, true, naj));
        net.addLayer(new BasicLayer(new ActivationSigmoid(), false, naj * 2));
        net.addLayer(new BasicLayer(new ActivationSigmoid(), false, naj * 4));
        net.addLayer(new BasicLayer(new ActivationSigmoid(), false, naj * 2));
        net.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        net.getStructure().finalizeStructure();
        net.reset();

        MLDataSet trainingSet = new BasicMLDataSet(Input, Ideal);
        final Backpropagation train = new Backpropagation(net, trainingSet);

        int epoch = 1;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error: " + train.getError());
            epoch++;
        } while (train.getError() > 0.00001);
        train.finishTraining();

        System.out.println("Network results: ");
        for (MLDataPair pair : trainingSet) {
            final MLData out = net.compute(pair.getInput());
            System.out.println(
                    idealno[((int) (out.getData(0) * 5 + 0.5) - 1) >= 0 ? (int) (out.getData(0) * 5 + 0.5) - 1 : 0]);
            // printOut(out);
        }
        Scanner in = new Scanner(System.in);
        System.out.println(
                "Upišite narudžbu s menija:\n(dozvoljena su samo engleska slova, ako su prisutna druga slova izlaz će biti neprecizan)");
        String rijec = in.nextLine();
        if (!rijec.equals("n"))
            do {
                double[] test = new double[naj];
                net.compute(norm(rijec), test);
                // printOutD(test);
                System.out.println(
                        "To je: " + idealno[((int) (test[0] * 5 + 0.5) - 1) >= 0 ? (int) (test[0] * 5 + 0.5) - 1 : 0]);
                System.out.println();
                System.out.println("Upišite narudžbu s menija ili \"n\" za kraj: ");
                rijec = in.nextLine();
            } while (!rijec.equals("n"));
        in.close();
        Encog.getInstance().shutdown();

    }
    public static double[] norm(String s) {
        double[] ret = new double[naj];
        for (int i = 0; i < naj; i++) {
            if (i < s.length())
                ret[i] = ((int) s.charAt(i) - 96) / 26.0;
            else
                ret[i] = 0.0;
        }
        return ret;
    }
}
