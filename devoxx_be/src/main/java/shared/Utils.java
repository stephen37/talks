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

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.segment.TextSegment;

import static dev.langchain4j.internal.Utils.getOrDefault;


public class Utils {
    public static final String OLLAMA_BASE_URL = "http://localhost:11434";

    private static final ObjectMapper objectMapper = new ObjectMapper();

    // public static boolean gradeRetrieval(String question, TextSegment document, ChatLanguageModel model) {
    //     String promptTemplate = """
    //         You are a grader assessing relevance of a retrieved document to a user question. 
    //         If the document contains keywords related to the user question, grade it as relevant. 
    //         It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

    //         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    //         Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

    //         Here is the retrieved document:
    //         {document}

    //         Here is the user question:
    //         {question}
    //         """;

    //     PromptTemplate prompt = PromptTemplate.from(promptTemplate);
    //     Map<String, Object> variables = new HashMap<>();
    //     variables.put("question", question);
    //     variables.put("document", document.text());
    //     String formattedPrompt = prompt.apply(variables);

    //     String response = model.generate(formattedPrompt);
    //     try {
    //         String jsonResponse = response.trim();
    //         Score score = objectMapper.readValue(jsonResponse, Score.class);
    //         return "yes".equalsIgnoreCase(score.score);
    //     } catch (Exception e) {
    //         e.printStackTrace();
    //         return false;
    //     }
    // }

    // private static class Score {
    //     public String score;
    // }

    public static ChatLanguageModel getLanguageModel() {
        return OllamaChatModel.builder()
                .baseUrl(OLLAMA_BASE_URL)
                .modelName("llama3.2")
                .build();
    }
    
    public static String getTavilyApiKey() {
       return System.getenv("TAVILY_API_KEY");
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
