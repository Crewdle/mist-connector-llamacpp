import { EventEmitter } from 'events';

import type { Llama, LlamaChatSession, LlamaModel, LlamaEmbeddingContext, LlamaContext, Token } from 'node-llama-cpp';

import type { IJob, IJobResult, IGenerativeAIWorkerConnector, IVectorDatabaseConnector, JobStatus, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';
import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerDocument } from './LlamacppGenerativeAIWorkerDocument';

/**
 * The Llamacpp machine learning connector.
 */
export class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
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
   * The documents.
   * @ignore
   */
  private documents: ILlamacppGenerativeAIWorkerDocument[] = [];

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
   * @param vectorDatabaseConnector The vector database connector constructor.
   */
  constructor(
    vectorDatabaseConnector: VectorDatabaseConnectorConstructor,
    private options?: ILlamacppGenerativeAIWorkerOptions,
  ) {
    this.vectorDatabase = new vectorDatabaseConnector();
  }

  /**
   * Initialize the machine learning model.
   * @param llmModel The path to the LLM model.
   * @param similarityModel The path to the similarity model.
   */
  async initialize(llmModel?: string, similarityModel?: string): Promise<void> {
    if (!llmModel && !this.options?.llmPath) {
      throw new Error('LLM model path not provided');
    }
    if (!similarityModel && !this.options?.similarityPath) {
      throw new Error('Similarity model path not provided');
    }
    const { getLlama } = await import('node-llama-cpp');
    this.engine = await getLlama();
    this.llmModel = await this.engine.loadModel({
      modelPath: llmModel ?? this.options?.llmPath ?? 'llm.gguf',
    });
    this.similarityModel = await this.engine.loadModel({
      modelPath: similarityModel ?? this.options?.similarityPath ?? 'similarity.gguf',
    });
  }

  /**
   * Add content to the machine learning model.
   * @param name The name of the content.
   * @param content The content to add.
   * @returns A promise that resolves when the content has been added.
   */
  async addContent(name: string, content: string): Promise<void> {
    if (!this.similarityModel) {
      throw new Error('Model not initialized');
    }

    this.embeddingContext = await this.similarityModel.createEmbeddingContext();

    const sentences = this.splitIntoSentences(content);
    this.documents.push({
      name,
      startIndex: this.sentences.length,
      length: sentences.length,
    });
    this.sentences.push(...sentences);

    const embeddings: number[][] = await Promise.all(sentences.map(sentence => this.getVector(sentence)));
    this.vectorDatabase.insert(embeddings);

    await this.embeddingContext.dispose();
    this.embeddingContext = undefined;
  }

  /**
   * Remove content from the machine learning model.
   * @param name The name of the content.
   */
  removeContent(name: string): void {
    const document = this.documents.find(document => document.name === name);
    if (!document) return;

    this.vectorDatabase.remove(Array.from({ length: document.length }, (_, i) => document.startIndex + i));
    this.sentences.splice(document.startIndex, document.length);
    this.documents = this.documents.filter(doc => doc !== document);
  }

  /**
   * Prompt the machine learning model.
   * @param prompt The prompt to use.
   * @returns An async generator that yields the responses.
   */
  async *processJob(job: IJob): AsyncGenerator<IJobResult> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    if (job.parameters.jobType !== 0) {
      return;
    }

    const { LlamaChatSession } = await import('node-llama-cpp');

    const prompt = job.parameters.prompt;
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
      yield {
        id: job.id,
        status: 'Completed' as JobStatus,
        result: {
          jobType: 0,
          output: this.llmModel.detokenize(token),
        },
      };
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
