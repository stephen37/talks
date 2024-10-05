package com.example;

import dev.langchain4j.chain.ConversationalChain;
import dev.langchain4j.model.openai.OpenAiChatModel;

public class LangchainDemo {
    public static void main(String[] args) {
        // Replace with your actual OpenAI API key
        String apiKey = "your-api-key-here";

        // Create an OpenAI chat model
        OpenAiChatModel model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .build();

        // Create a conversational chain
        ConversationalChain chain = ConversationalChain.builder()
                .model(model)
                .build();

        // Start a conversation
        String response = chain.execute("Hello, how are you?");
        System.out.println("AI: " + response);

        // Continue the conversation
        response = chain.execute("What's the weather like today?");
        System.out.println("AI: " + response);
    }
}