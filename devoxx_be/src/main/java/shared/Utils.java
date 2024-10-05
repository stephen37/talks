package shared;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.Paths;
import java.util.Scanner;

import static dev.langchain4j.internal.Utils.getOrDefault;

public class Utils {

    private static final Dotenv dotenv = Dotenv.configure().ignoreIfMissing().load();

    public static String getEnv(String key) {
        return dotenv.get(key);
    }

    public static String getEnv(String key, String defaultValue) {
        return dotenv.get(key, defaultValue);
    }

    public static final String OLLAMA_BASE_URL = getEnv("OLLAMA_BASE_URL", "http://localhost:11434");

    public static ChatLanguageModel createOllamaChatModel() {
    return OllamaChatModel.builder()
            .baseUrl(OLLAMA_BASE_URL)
            .modelName("llama3.2")
            .build();
}

    public static void startConversationWith(Assistant assistant) {
        Logger log = LoggerFactory.getLogger(Assistant.class);
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                log.info("==================================================");
                log.info("User: ");
                String userQuery = scanner.nextLine();
                log.info("==================================================");

                if ("exit".equalsIgnoreCase(userQuery)) {
                    break;
                }

                String agentAnswer = assistant.answer(userQuery);
                log.info("==================================================");
                log.info("Assistant: " + agentAnswer);
            }
        }
    }

    public static PathMatcher glob(String glob) {
        return FileSystems.getDefault().getPathMatcher("glob:" + glob);
    }

    public static Path toPath(String relativePath) {
        try {
            URL fileUrl = Utils.class.getClassLoader().getResource(relativePath);
            return Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
