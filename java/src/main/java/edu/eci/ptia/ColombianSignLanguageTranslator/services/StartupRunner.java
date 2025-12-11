package edu.eci.ptia.ColombianSignLanguageTranslator.services;

import edu.eci.ptia.ColombianSignLanguageTranslator.services.CameraService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class StartupRunner implements CommandLineRunner {

    private final CameraService cameraService;

    public StartupRunner(CameraService cameraService) {
        this.cameraService = cameraService;
    }

    @Override
    public void run(String... args) {
        System.out.println("📸 Abriendo cámara...");
        cameraService.startCamera();
    }
}
