# Crewdle Mist llama.cpp Generative AI Connector

## Introduction

The Crewdle Mist llama.cpp Generative AI Connector is a solution designed to seamlessly run Large Language Models (LLMs) within your applications. This connector enables applications to harness the power of generative AI, providing robust and high-performance model execution capabilities. With its straightforward integration and reliable operation, it requires the use of a Crewdle Mist vector database connector to ensure efficient indexing and searching of content. It's an ideal choice for developers seeking to implement scalable and sophisticated AI-driven functionalities within their ecosystem.

## Getting Started

Before diving in, ensure you have installed the [Crewdle Mist SDK](https://www.npmjs.com/package/@crewdle/web-sdk).

## Installation

```bash
npm install @crewdle/mist-connector-llamacpp
```

## Usage

```TypeScript
const { getLlamacppGenerativeAIWorkerConnector } = await import('@crewdle/mist-connector-llamacpp');

sdk = await SDK.getInstance(config.vendorId, config.accessToken, {
  generativeAIWorkerConnector: getLlamacppGenerativeAIWorkerConnector({
    llmPath: '/models/llm.gguf',
    similarityPath: '/models/similarity.gguf',
  }),
}, config.secretKey);
```

## Need Help?

Reach out to support@crewdle.com or raise an issue in our repository for any assistance.

## Join Our Community

For an engaging discussion about your specific use cases or to connect with fellow developers, we invite you to join our Discord community. Follow this link to become a part of our vibrant group: [Join us on Discord](https://discord.gg/XJ3scBYX).
