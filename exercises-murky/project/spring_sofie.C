#include "TMVA/RSofieReader.hxx"
#include <vector>
#include <iostream>
#include <cmath>

void spring_sofie()
{
    TMVA::Experimental::RSofieReader reader("spring_model.onnx");

    float k = 10.0;
    float m = 1.0;

    float w = sqrt(k/m);

    std::cout << "t   AI_prediction   True_value\n";

    for(float t = 0; t <= 3; t += 0.25)
    {
        std::vector<float> input = {t};

        auto output = reader.Compute(input);

        float true_val = cos(w*t);

        std::cout << t
                  << "   "
                  << output[0]
                  << "   "
                  << true_val
                  << std::endl;
    }
}