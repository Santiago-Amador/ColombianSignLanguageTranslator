package edu.eci.ptia.ColombianSignLanguageTranslator.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;
import edu.eci.ptia.ColombianSignLanguageTranslator.config.AppConfig;
@Getter
@Component
@Slf4j
public class ONNXLoader {

    private final OrtEnvironment env;
    private final OrtSession session;

    @Autowired
    public ONNXLoader(AppConfig appConfig, ResourceLoader resourceLoader) throws Exception {
        log.info("Loading model CNN from: {}", appConfig.getModelPath());

        Resource modelResource = resourceLoader.getResource(appConfig.getModelPath());

        if (!modelResource.exists()){
            throw new RuntimeException("No se encontro el modelo en: " + appConfig.getModelPath());
        }
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelResource.getFile().getAbsolutePath());

        log.info("ONNX model loaded successfully.");
        log.info("   └── Inputs : {}", session.getInputNames());
        log.info("   └── Outputs: {}", session.getOutputNames());
    }

}
