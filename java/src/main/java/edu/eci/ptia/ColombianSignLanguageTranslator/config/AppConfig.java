package edu.eci.ptia.ColombianSignLanguageTranslator.config;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
@Configuration
@Getter
public class AppConfig {
    @Value("${model.path:classpath:model/model.onnx}")
    private String modelPath;

    @Value("${classes.path:classpath:model/classes.txt}")
    private String classesPath;

    //Tamaño de las imagenes en el entrenamiento del modelo
    private final int imgWidth = 128;
    private final int imgLength = 128;
    private final int channels = 3;

    private final int cameraIndex = 0;



}
