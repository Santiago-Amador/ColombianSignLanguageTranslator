package edu.eci.ptia.ColombianSignLanguageTranslator.services;

import edu.eci.ptia.ColombianSignLanguageTranslator.config.AppConfig;
import edu.eci.ptia.ColombianSignLanguageTranslator.services.SignPredictionService;
import org.springframework.stereotype.Service;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.global.opencv_highgui;
import org.bytedeco.opencv.global.opencv_imgproc;

@Service
public class CameraService {

    private final AppConfig config;
    private final SignPredictionService predictor;

    public CameraService(AppConfig config, SignPredictionService predictor) {
        this.config = config;
        this.predictor = predictor;
    }

    public void startCamera() {

        VideoCapture cap = new VideoCapture(config.getCameraIndex());

        if (!cap.isOpened()) {
            throw new RuntimeException("No se pudo abrir la cámara");
        }

        Mat frame = new Mat();

        while (true) {

            if (!cap.read(frame)) {
                continue;
            }


            String prediction;
            try {
                prediction = predictor.predict(frame);
            } catch (Exception e) {
                prediction = "Error";
            }


            opencv_imgproc.putText(
                    frame,
                    "Prediccion: " + prediction,
                    new Point(10, 30),
                    opencv_imgproc.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    new Scalar(0, 255, 0, 0) // Verde
            );


            opencv_highgui.imshow("Colombian Sign Language - Live", frame);

            if (opencv_highgui.waitKey(1) == 27) {
                break;
            }
        }

        cap.release();
        opencv_highgui.destroyAllWindows();
    }
}
