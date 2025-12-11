package edu.eci.ptia.ColombianSignLanguageTranslator.services;
import edu.eci.ptia.ColombianSignLanguageTranslator.config.AppConfig;
import edu.eci.ptia.ColombianSignLanguageTranslator.model.ClassLoader;
import edu.eci.ptia.ColombianSignLanguageTranslator.model.ImagePreprocessor;
import edu.eci.ptia.ColombianSignLanguageTranslator.model.ONNXLoader;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Service;
import ai.onnxruntime.*;

import java.util.Collections;
@RequiredArgsConstructor
@Service
@Slf4j
public class SignPredictionService {
    private final ONNXLoader onnxLoader;
    private final ClassLoader classLoader;
    private final ImagePreprocessor imagePreprocessor;
    private final AppConfig config;

    public String predict(Mat frame){
        try {

            float[] inputData = imagePreprocessor.preprocess(frame);


            float[][][][] input4d = new float[1][config.getImgLength()][config.getImgWidth()][config.getChannels()];

            int idx = 0;
            for (int i = 0; i < config.getImgLength(); i++) {
                for (int j = 0; j < config.getImgWidth(); j++) {
                    for (int c = 0; c < config.getChannels(); c++) {
                        input4d[0][i][j][c] = inputData[idx++];
                    }
                }
            }

            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    onnxLoader.getEnv(),
                    input4d
            );


            OrtSession.Result result = onnxLoader.getSession().run(
                    Collections.singletonMap(
                            onnxLoader.getInputName(),
                            inputTensor
                    )
            );

            float[][] output = (float[][]) result.get(0).getValue();


            int index = argMax(output[0]);
            String predictedClass = classLoader.getClasses().get(index);

            log.info("Predicción: {} (índice: {})", predictedClass, index);

            return predictedClass;

        } catch (Exception e) {
            log.error("Error en la predicción", e);
            return "?";
        }
    }

    private int argMax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxIndex = i;
                maxValue = array[i];
            }
        }
        return maxIndex;
    }
}
