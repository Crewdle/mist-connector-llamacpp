import { EventEmitter } from 'events';

import { Llama, getLlama, LlamaChatSession, LlamaModel, LlamaEmbeddingContext, LlamaContext, Token } from 'node-llama-cpp';

import { IMachineLearningConnector, IVectorDatabaseConnector, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';

/**
 * The Llamacpp machine learning connector.
 * @category Connector
 */
export class LlamacppMachineLearningConnector implements IMachineLearningConnector {
  /**
   * The vector database connector.
   * @ignore
   */
  private vectorDatabase: IVectorDatabaseConnector;

  /**
   * The instructions.
   * @ignore
   */
  private instructions = 'Keep the answer short and concise.'

  /**
   * The Llama engine.
   * @ignore
   */
  private engine?: Llama;

  /**
   * The model to use for language modeling.
   * @ignore
   */
  private llmModel?: LlamaModel;

  /**
   * The model to use for sentence similarity.
   * @ignore
   */
  private similarityModel?: LlamaModel;

  /**
   * The sentences.
   * @ignore
   */
  private sentences: string[] = [];

  /**
   * The embedding context.
   * @ignore
   */
  private embeddingContext?: LlamaEmbeddingContext;

  /**
   * The chat context.
   * @ignore
   */
  private chatContext?: LlamaContext;

  /**
   * The chat session.
   * @ignore
   */
  private chatSession?: LlamaChatSession;

  /**
   * The constructor.
   * @param vectorDatabaseConstructor The vector database connector constructor.
   */
  constructor(
    vectorDatabaseConstructor: VectorDatabaseConnectorConstructor,
  ) {
    this.vectorDatabase = new vectorDatabaseConstructor();
  }

  /**
   * Initialize the machine learning model.
   * @param llmModel The path to the LLM model.
   * @param similarityModel The path to the similarity model.
   */
  async initialize(llmModel: string, similarityModel: string): Promise<void> {
    this.engine = await getLlama();
    this.llmModel = await this.engine.loadModel({
      modelPath: llmModel,
    });
    this.similarityModel = await this.engine.loadModel({
      modelPath: similarityModel,
    });
  }

  /**
   * Add content to the machine learning model.
   * @param content The content to add.
   * @returns A promise that resolves when the content has been added.
   */
  async addContent(content: string): Promise<void> {
    if (!this.similarityModel) {
      throw new Error('Model not initialized');
    }

    this.embeddingContext = await this.similarityModel.createEmbeddingContext();

    const sentences = this.splitIntoSentences(content);
    this.sentences.push(...sentences);

    const embeddings: number[][] = await Promise.all(sentences.map(sentence => this.getVector(sentence)));
    this.vectorDatabase.insert(embeddings);

    await this.embeddingContext.dispose();
    this.embeddingContext = undefined;
  }

  /**
   * Prompt the machine learning model.
   * @param prompt The prompt to use.
   * @returns An async generator that yields the responses.
   */
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

  /**
   * Get the context for a prompt.
   * @param prompt The prompt.
   * @returns A promise that resolves to the context.
   * @ignore
   */
  private async getContext(prompt: string): Promise<string> {
    if (!this.similarityModel) {
      throw new Error('Model not initialized');
    }

    this.embeddingContext = await this.similarityModel.createEmbeddingContext();
    const embedding: number[] = await this.getVector(prompt);
    const context = this.vectorDatabase.search(embedding, 5).map(index => this.sentences.slice(index - 2, index + 3).join(' '));
    await this.embeddingContext.dispose();

    return context.join(' ');
  }

  /**
   * Get the vector for some content.
   * @param content The content.
   * @returns A promise that resolves to the embedding.
   * @ignore
   */
  private async getVector(content: string): Promise<number[]> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    if (!this.embeddingContext) {
      throw new Error('Embedding context not initialized');
    }

    return (await this.embeddingContext.getEmbeddingFor(content)).vector;
  }

  /**
   * Split text into sentences.
   * @param text The text to split.
   * @returns The sentences.
   * @ignore
   */
  private splitIntoSentences(text: string): string[] {
    return text.match(/[^\.!\?]+[\.!\?]+/g)?.filter(sentence => sentence.trim().length > 0) || [];
  }
}
