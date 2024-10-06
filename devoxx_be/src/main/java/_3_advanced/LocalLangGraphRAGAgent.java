package _3_advanced;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.model.chat.ChatLanguageModel;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

import static shared.Utils.OLLAMA_BASE_URL;

public class LocalLangGraphRAGAgent {

    private final ChatLanguageModel chatModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingModel embeddingModel;

    public LocalLangGraphRAGAgent() {
        this.chatModel = OllamaChatModel.builder()
                .baseUrl(OLLAMA_BASE_URL)
                .modelName("llama3.2")
                .build();

        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.embeddingStore = createMilvusEmbeddingStore();
    }

    private EmbeddingStore<TextSegment> createMilvusEmbeddingStore() {
        return MilvusEmbeddingStore.builder()
                .host("localhost")
                .port(19530)
                .collectionName("rag_milvus")
                .dimension(384)
                .build();
    }

    public CompletableFuture<String> processQuery(String question) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                String routeDecision = routeQuestion(question);
                switch (routeDecision) {
                    case "vectorstore":
                        return handleVectorStore(question);
                    default:
                        return "Unable to process the question.";
                }
            } catch (Exception e) {
                throw new RuntimeException("Error processing query", e);
            }
        });
    }

    private void loadAndEmbedDocuments(List<String> texts) {
        for (String text : texts) {
            TextSegment segment = TextSegment.from(text);
            Embedding embedding = embeddingModel.embed(segment).content();
            embeddingStore.add(embedding, segment);
        }
    }

   private String handleVectorStore(String question) {
        Embedding queryEmbedding = embeddingModel.embed(question).content();
        List<EmbeddingMatch<TextSegment>> relevantSegments = embeddingStore.findRelevant(queryEmbedding, 2);
        List<TextSegment> segments = relevantSegments.stream()
                .map(EmbeddingMatch::embedded)
                .toList();
        return generate(question, segments);
    }

    private String routeQuestion(String question) {
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Route this question to either 'vectorstore' or 'websearch': {{question}}"
        );
        Prompt prompt = promptTemplate.apply(Map.of("question", question));
        return chatModel.generate(prompt.toUserMessage()).content().text();
    }

    private String generate(String question, List<TextSegment> context) {
        String contextString = context.stream()
                .map(TextSegment::text)
                .reduce("", (a, b) -> a + "\n" + b);

        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer this question based on the given context:\n" +
                        "Question: {{question}}\n" +
                        "Context: {{context}}"
        );
        Prompt prompt = promptTemplate.apply(Map.of("question", question, "context", contextString));
        return chatModel.generate(prompt.toUserMessage()).content().text();
    }
    public static void main(String[] args) {
        LocalLangGraphRAGAgent agent = new LocalLangGraphRAGAgent();
        
        // Load and embed some sample data
        List<String> sampleTexts = List.of(
            "Emmanuel Macron visited Germany last week for diplomatic talks.",
            "The weather in Paris has been unusually warm this summer.",
            "France and Germany are key members of the European Union."
        );
        agent.loadAndEmbedDocuments(sampleTexts);

        String question = "Did Emmanuel Macron visit Germany recently?";
        agent.processQuery(question).thenAccept(System.out::println);
    }
}