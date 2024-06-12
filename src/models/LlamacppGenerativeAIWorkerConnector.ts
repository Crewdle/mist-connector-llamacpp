import { EventEmitter } from 'events';

import type { Llama, LlamaChatSession, LlamaModel, LlamaEmbeddingContext, LlamaContext, Token } from 'node-llama-cpp';

import type { IGenerativeAIWorkerConnector, IJobParametersAI, IJobRequest, IJobResultAI, IVectorDatabaseConnector, JobResponse, JobStatus, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';

import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerDocument } from './LlamacppGenerativeAIWorkerDocument';

const SENTENCE_MAX_LENGTH = 512;

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
  private instructions = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it will use concepts stored in memory that have very similar weights.`;

  /**
   * The maximum number of tokens to generate.
   * @ignore
   */
  private maxTokens = 1024;

  /**
   * The temperature.
   * @ignore
   */
  private temperature = 1;

  /**
   * The number of contents to include in the context.
   * @ignore
   */
  private maxContents = 5;

  /**
   * The number of sentences to include in one content.
   * @ignore
   */
  private maxSentences = 6;

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
    if (options?.instructions) {
      this.instructions = options.instructions;
    }
    if (options?.maxTokens) {
      this.maxTokens = options.maxTokens;
    }
    if (options?.temperature) {
      this.temperature = options.temperature;
    }
    if (options?.maxContents) {
      this.maxContents = options.maxContents;
    }
    if (options?.maxSentences) {
      this.maxSentences = options.maxSentences;
    }
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

    const document = this.documents.find(document => document.name === name);
    if (document) {
      this.removeContent(name);
    }

    this.embeddingContext = await this.similarityModel.createEmbeddingContext();

    const sentences = this.splitIntoSentences(content);
    const embeddings: number[][] = [];
    const sentencesToAdd: string[] = [];
    for (let i = 0; i < sentences.length; i++) {
      try {
        for (let j = 0; j < sentences[i].length; j += SENTENCE_MAX_LENGTH) {
          const embedding: number[] = await this.getVector(sentences[i].slice(j, j + SENTENCE_MAX_LENGTH));
          embeddings.push(embedding);
          sentencesToAdd.push(sentences[i].slice(j, j + SENTENCE_MAX_LENGTH));
        }
      } catch (e) {
        console.error(e);
      }
    }
    if (embeddings.length > 0) {
      this.documents.push({
        name,
        startIndex: this.sentences.length,
        length: sentencesToAdd.length,
      });
      this.sentences.push(...sentencesToAdd);
      this.vectorDatabase.insert(embeddings);
    }

    await this.embeddingContext.dispose();
    this.embeddingContext = undefined;

    console.log(`Added content: ${name}`);
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
   * Process a job.
   * @param job The job to process.
   * @returns A promise that resolves with the job result.
   */
  async processJob(job: IJobRequest<IJobParametersAI>): Promise<JobResponse<IJobResultAI>> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    if (!this.chatContext) {
      this.chatContext = await this.llmModel.createContext();
    }

    if (!this.chatSession) {
      const { LlamaChatSession } = await import('node-llama-cpp');
      this.chatSession = new LlamaChatSession({
        contextSequence: this.chatContext.getSequence(),
      });
    }

    const prompt = await this.getPrompt(job);

    const output = await this.chatSession.prompt(prompt, {
      maxTokens: this.maxTokens,
      temperature: this.temperature,
    });

    const inputTokens = this.llmModel.tokenize(prompt).length;
    const outputTokens = this.llmModel.tokenize(output).length;

    this.chatSession.dispose();
    this.chatSession = undefined;
    this.chatContext.dispose();
    this.chatContext = undefined;

    return {
      id: job.id,
      status: 'completed' as JobStatus.Completed,
      result: {
        output,
        inputTokens,
        outputTokens,
      },
    };
  }

  /**
   * Stream a job.
   * @param job The job to stream.
   * @returns An async generator that yields the responses.
   */
  async *processJobStream(job: IJobRequest<IJobParametersAI>): AsyncGenerator<JobResponse<IJobResultAI>> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    if (!this.chatContext) {
      this.chatContext = await this.llmModel.createContext();
    }

    if (!this.chatSession) {
      const { LlamaChatSession } = await import('node-llama-cpp');
      this.chatSession = new LlamaChatSession({
        contextSequence: this.chatContext.getSequence(),
      });
    }

    const tokenEmitter = new EventEmitter();

    const prompt = await this.getPrompt(job);
    
    const inputTokens = this.llmModel.tokenize(prompt).length;
    let outputTokens = 0;

    this.chatSession.prompt(prompt, {
      maxTokens: this.maxTokens,
      temperature: this.temperature,
      onToken: async (token) => {
        tokenEmitter.emit('token', token);
      }
    }).then(() => {
      tokenEmitter.emit('token', undefined);
    }).catch(e => {});

    while (true) {
      const token = await new Promise<Token[]>((resolve) => tokenEmitter.once('token', resolve));
      if (!token || this.llmModel.detokenize(token).indexOf('<|end|>') !== -1) {
        break;
      }
      outputTokens += token.length;
      yield {
        id: job.id,
        status: 'partial' as JobStatus.Partial,
        result: {
          output: this.llmModel.detokenize(token),
          inputTokens,
          outputTokens,
        },
      };
    }

    this.chatSession.dispose();
    this.chatSession = undefined;
    this.chatContext.dispose();
    this.chatContext = undefined;
  }

  private async getPrompt(job: IJobRequest<IJobParametersAI>): Promise<string> {
    if (!this.llmModel) {
      throw new Error('Model not initialized');
    }

    const { useRAG, prompt, instructions, context } = job.parameters;

    let finalPrompt = `${instructions ?? this.instructions}\n\n`;

    if (useRAG === undefined || useRAG === true) {
      let ragContext = '';
      ragContext = await this.getContext(prompt);
      if (ragContext !== '') {
        finalPrompt += `Relevant Information:\n\n${ragContext}\n\n`;
      }
    }

    finalPrompt += `Conversation:\n`;
    if (context) {
      let conversation = '';
      let conversationLength = this.llmModel.tokenize(finalPrompt).length + this.llmModel.tokenize(`Human: ${prompt}\nAI:`).length;
      for (let i = context.length - 1; i >= 0; i--) {
        const { source, message } = context[i];
        if (conversationLength + this.llmModel.tokenize(message).length > this.llmModel.trainContextSize) {
          break;
        }
        conversation = `${source === 'ai' ? 'AI' : 'Human'}: ${message}\n${conversation}`;
        conversationLength += this.llmModel.tokenize(message).length;
      }
      finalPrompt += conversation;
    }

    finalPrompt += `Human: ${prompt}\nAI:`;

    return finalPrompt;
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
    const context = this.vectorDatabase.search(embedding, this.maxContents).map(index => {
      let context = '';
      const document = this.documents.find(document => document.startIndex <= index && index < document.startIndex + document.length);
      if (document) {
        context += `From ${document.name}:\n`
      }
      context += this.sentences.slice(index - (this.maxSentences / 2 - 1), index + (this.maxSentences / 2)).join(' ');
      return context;
    });
    await this.embeddingContext.dispose();

    return context.join("\n");
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
