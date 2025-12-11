package edu.eci.ptia.ColombianSignLanguageTranslator.model;

import edu.eci.ptia.ColombianSignLanguageTranslator.config.AppConfig;
import lombok.RequiredArgsConstructor;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class ImagePreprocessor {

    private final AppConfig config;

    public float[] preprocess(Mat frame) {
        
        Mat rgb = new Mat();
        opencv_imgproc.cvtColor(frame, rgb, opencv_imgproc.COLOR_BGR2RGB);

        
        Mat resized = new Mat();
        opencv_imgproc.resize(rgb, resized, new Size(config.getImgWidth(), config.getImgLength()));


        UByteIndexer indexer = resized.createIndexer();

        
        int width = config.getImgWidth();
        int height = config.getImgLength();
        int channels = config.getChannels();

        float[] input = new float[width * height * channels];

        int idx = 0;

        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                
                double r = indexer.get(y, x, 0);
                double g = indexer.get(y, x, 1);
                double b = indexer.get(y, x, 2);

                
                input[idx++] = (float) (r / 255.0);
                input[idx++] = (float) (g / 255.0);
                input[idx++] = (float) (b / 255.0);
            }
        }

        return input;
    }
}
