package edu.eci.ptia.ColombianSignLanguageTranslator.model;
import edu.eci.ptia.ColombianSignLanguageTranslator.config.AppConfig;
import lombok.Getter;
import org.springframework.stereotype.Component;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
@Getter
@Component
public class ClassLoader {
    private final List<String> classes = new ArrayList<>();

    public ClassLoader(AppConfig appConfig, ResourceLoader resourceLoader) throws Exception{
        Resource resource = resourceLoader.getResource(appConfig.getClassesPath());

        if(!resource.exists()){
            throw new RuntimeException("Classes not found at " + appConfig.getClassesPath());
        }

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {

            String line;
            while ((line = br.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    classes.add(line.trim());
                }
            }
        }
        System.out.println("Loaded: " + classes.size() + " Classes: " + classes);
    }

}
