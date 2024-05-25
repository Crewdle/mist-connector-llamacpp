import { EventEmitter } from 'events';

import { Llama, getLlama, LlamaChatSession, LlamaModel, LlamaEmbeddingContext, LlamaContext, Token } from 'node-llama-cpp';

import { IMachineLearningConnector, IVectorDatabaseConnector, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';

export class LlamacppMachineLearningConnector implements IMachineLearningConnector {
  private vectorDatabase: IVectorDatabaseConnector;
  private instructions = 'Keep the answer short and concise.'
  private engine?: Llama;
  private llmModel?: LlamaModel;
  private similarityModel?: LlamaModel;
  private sentences: string[] = [];
  private embeddingContext?: LlamaEmbeddingContext;
  private chatContext?: LlamaContext;
  private chatSession?: LlamaChatSession;

  constructor(
    vectorDatabaseConstructor: VectorDatabaseConnectorConstructor,
  ) {
    this.vectorDatabase = new vectorDatabaseConstructor();
  }

  async initialize(llmModel: string, similarityModel: string): Promise<void> {
    this.engine = await getLlama();
    this.llmModel = await this.engine.loadModel({
      modelPath: llmModel,
    });
    this.similarityModel = await this.engine.loadModel({
      modelPath: similarityModel,
    });
  }

  async addContent(content: string): Promise<void> {
    if (!this.similarityModel) {
      throw new Error('Model not initialized');
    }

    this.embeddingContext = await this.similarityModel.createEmbeddingContext();

    const sentences = this.splitIntoSentences(content);
    this.sentences.push(...sentences);

    const embeddings: number[][] = await Promise.all(sentences.map(sentence => this.getEmbedding(sentence)));
    this.vectorDatabase.insert(embeddings);

    this.embeddingContext.dispose();
    this.embeddingContext = undefined;
  }

  async *prompt(prompt: string): AsyncGenerator<string> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    const context = await this.getContext(prompt);

    if (!this.chatContext) {
      this.chatContext = await this.llmModel.createContext();
    }

    if (!this.chatSession) {
      this.chatSession = new LlamaChatSession({
        contextSequence: this.chatContext.getSequence(),
      });
    }

    const tokenEmitter = new EventEmitter();

    this.chatSession.prompt(`Instructions: ${this.instructions}, Context: ${context}, Prompt: ${prompt}`, {
      maxTokens: 1024,
      temperature: 0.5,
      onToken: async (token) => {
        tokenEmitter.emit('token', token);
      }
    });

    while (true) {
      const token = await new Promise<Token[]>((resolve) => tokenEmitter.once('token', resolve));
      if (!token) break;
      yield this.llmModel.detokenize(token);
    }
  }

  private async getContext(prompt: string): Promise<string[]> {
    if (!this.similarityModel) {
      throw new Error('Model not initialized');
    }

    this.embeddingContext = await this.similarityModel.createEmbeddingContext();
    const embedding: number[] = await this.getEmbedding(prompt);
    const context = this.vectorDatabase.search(embedding, 5).map(index => this.sentences.slice(index - 2, index + 3).join(' '));
    this.embeddingContext.dispose();

    return context;
  }

  private async getEmbedding(content: string): Promise<number[]> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    if (!this.embeddingContext) {
      throw new Error('Embedding context not initialized');
    }

    return (await this.embeddingContext.getEmbeddingFor(content)).vector;
  }

  private splitIntoSentences(text: string): string[] {
    return text.match(/[^\.!\?]+[\.!\?]+/g)?.filter(sentence => sentence.trim().length > 0) || [];
  }
}
