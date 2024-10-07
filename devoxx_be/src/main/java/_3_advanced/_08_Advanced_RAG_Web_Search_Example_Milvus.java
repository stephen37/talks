package _3_advanced;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.huggingface.HuggingFaceEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import shared.Assistant;

import java.util.List;
import java.time.Duration;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class _08_Advanced_RAG_Web_Search_Example_Milvus {

        public static void main(String[] args) {
                Assistant assistant = createAssistant();
                startConversationWith(assistant);
        }

        private static Assistant createAssistant() {
                EmbeddingModel embeddingModel = createLocalEmbeddingModel();

                EmbeddingStore<TextSegment> embeddingStore = createMilvusEmbeddingStore();
                List<String> documentPaths = List.of(
                                "documents/miles-of-smiles-terms-of-use.txt",
                                "documents/biography-of-john-doe.txt");

                embedMultipleDocuments(documentPaths, embeddingModel, embeddingStore);

                ContentRetriever embeddingStoreContentRetriever = EmbeddingStoreContentRetriever.builder()
                                .embeddingStore(embeddingStore)
                                .embeddingModel(embeddingModel)
                                .maxResults(3)
                                .build();

                WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                                .apiKey(getTavilyApiKey())
                                .build();

                ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                                .webSearchEngine(webSearchEngine)
                                .maxResults(3)
                                .build();

                QueryRouter queryRouter = new DefaultQueryRouter(embeddingStoreContentRetriever,
                                webSearchContentRetriever);

                RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                                .queryRouter(queryRouter)
                                .build();

                ChatLanguageModel model = OllamaChatModel.builder()
                                .baseUrl(OLLAMA_BASE_URL)
                                .modelName("llama3.2")
                                .build();

                return AiServices.builder(Assistant.class)
                                .chatLanguageModel(model)
                                .retrievalAugmentor(retrievalAugmentor)
                                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                                .build();
        }

        private static EmbeddingModel createLocalEmbeddingModel() {
                return HuggingFaceEmbeddingModel.builder()
                                .accessToken(System.getenv("HF_API_KEY"))
                                .modelId("sentence-transformers/all-MiniLM-L6-v2")
                                .waitForModel(true)
                                .timeout(Duration.ofSeconds(120))
                                .build();
        }

        private static EmbeddingStore<TextSegment> createMilvusEmbeddingStore() {
                return MilvusEmbeddingStore.builder()
                                .host("localhost")
                                .port(19530)
                                .collectionName("rag_demo")
                                .dimension(384)
                                .build();
        }

        private static void embedMultipleDocuments(List<String> documentPaths, EmbeddingModel embeddingModel,
                        EmbeddingStore<TextSegment> embeddingStore) {
                DocumentParser documentParser = new TextDocumentParser();
                DocumentSplitter splitter = DocumentSplitters.recursive(300, 50);

                for (String path : documentPaths) {
                        Document document = loadDocument(toPath(path), documentParser);
                        List<TextSegment> segments = splitter.split(document);
                        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
                        embeddingStore.addAll(embeddings, segments);
                }
        }
}